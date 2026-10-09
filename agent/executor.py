"""Agent 执行器：运行状态图、记录思考/工具调用日志、限制最大步数、防死循环。

提供两条执行路径：
- `run` / `run_stream`：非流式（fallback），一次返回完整答案；
- `run_stream_typewriter`：逐 token 流式（SSE），若网关不支持则自动降级到非流式路径。
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import Settings
from app.exceptions import LLMError

logger = logging.getLogger(__name__)


class AgentExecutor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._graph = None
        self._streaming_graph = None
        # 记录流式路径是否已确认可用（避免对不支持 SSE 的网关反复重试）
        self._streaming_known_broken = False

    @property
    def graph(self):
        if self._graph is None:
            from agent.graph import build_agent_graph

            self._graph = build_agent_graph(self.settings)
        return self._graph

    @property
    def streaming_graph(self):
        if self._streaming_graph is None:
            from agent.graph import build_agent_graph_streaming

            self._streaming_graph = build_agent_graph_streaming(self.settings)
        return self._streaming_graph

    def reset_graph(self) -> None:
        """丢弃已编译的状态图，下次调用按当前配置重建。

        状态图在构建时把 ChatOpenAI（含模型名与端点）绑进了节点闭包，
        切换模型后不重建的话 Agent 仍会打旧模型。
        同时重置流式可用性标记——新端点可能支持/不支持 SSE，需重新探测。
        """
        self._graph = None
        self._streaming_graph = None
        self._streaming_known_broken = False

    def list_tools(self) -> list[dict]:
        from agent.tools import get_tools

        return [{"name": t.name, "description": t.description} for t in get_tools(self.settings)]

    def run(self, message: str, history: list[dict] | None = None) -> dict[str, Any]:
        """非流式执行（兼容旧调用）：汇总流式事件，返回 answer + steps。"""
        steps: list[dict] = []
        answer = ""
        for event in self.run_stream(message, history):
            if event["type"] == "step":
                steps.append(event["step"])
            elif event["type"] == "answer":
                answer += event["content"]
        return {"answer": answer, "steps": steps}

    def run_stream(self, message: str, history: list[dict] | None = None):
        """执行 Agent，逐步产出事件 dict。

        - {"type": "step", "step": {...}}     工具调用 / 观察结果（每步完成即推送）
        - {"type": "answer", "content": str}  最终答案

        说明：答案是整段推送，不按 token 分块。图内 LLM 未开启 streaming，
        且部分 OpenAI 兼容网关（自建中转）不支持 SSE —— 开启会让这些端点整体不可用，
        因此这里不做逐 token 推送，前端按整段追加渲染。
        """
        messages = self._build_messages(message, history)
        # 一次完整 ReAct 步包含 agent + tools 两个节点，recursion_limit 即步数硬上限
        recursion_limit = self.settings.agent_max_steps * 2 + 2
        final_answer = ""

        try:
            stream = self.graph.stream(
                {"messages": messages},
                config={"recursion_limit": recursion_limit},
                stream_mode="updates",
            )
            for data in stream:
                for node_name, node_output in data.items():
                    for msg in node_output.get("messages", []):
                        if (
                            isinstance(msg, AIMessage)
                            and msg.content
                            and not msg.tool_calls
                        ):
                            # 无工具调用的 AIMessage 就是最终答案本身，
                            # 不能再作为 thought 步骤推送，否则答案会出现两次
                            final_answer = msg.content
                            yield {"type": "answer", "content": final_answer}
                            continue
                        recorded = self._record_message(node_name, msg)
                        if recorded:
                            yield {"type": "step", "step": recorded}
        except Exception as exc:
            # 递归超限说明步数用尽；其它错误（鉴权/网络）包装成 LLMError
            if self._is_recursion_error(exc):
                final_answer = final_answer or "已达到最大推理步数限制，任务被迫终止，请简化问题后重试。"
                logger.warning("Agent 超过最大步数 %d", self.settings.agent_max_steps)
                yield {"type": "answer", "content": final_answer}
            else:
                raise LLMError(f"Agent 执行失败: {exc}") from exc

        if not final_answer:
            final_answer = "Agent 未能生成有效回答，请检查模型配置或稍后重试。"
            yield {"type": "answer", "content": final_answer}

    def run_stream_typewriter(self, message: str, history: list[dict] | None = None):
        """逐 token 流式执行 Agent（打字机效果），网关不支持 SSE 时自动降级。

        优先用流式图（stream_mode=["updates","messages"]）按 token 分块产出
        answer 事件。若当前网关不支持 SSE（自建中转/LM Studio 常这样），
        首次调用会抛错，此时把 `_streaming_known_broken` 置位并整段降级到
        非流式 `run_stream`，功能不受影响。

        - {"type": "step", "step": {...}}      工具调用 / 观察结果
        - {"type": "answer", "content": str}   答案分块（流式）或整段（降级）
        - {"type": "streaming_fallback", "reason": str}  降级通知（客户端可据此提示"非流式"）

        注意：本方法在同步上下文（FastAPI `def` 路由）里调用；流式图的
        `astream` 需要一个事件循环，故在后台线程里 `asyncio.run`，
        主线程用 50ms 短轮询把已产出的 token / 步骤转发出去，保证 SSE
        的 `event_stream` 持续生成、不阻塞。
        """
        if self._streaming_known_broken:
            yield from self.run_stream(message, history)
            return

        messages = self._build_messages(message, history)
        recursion_limit = self.settings.agent_max_steps * 2 + 2
        box: dict = {"chunks": [], "steps": [], "done": False, "error": None, "step_dirty": False}
        # 待转发的答案分块、步骤（主线程从 box 取出后清空）
        pending_chunks: list[str] = []
        pending_steps: list[dict] = []

        def _drain() -> None:
            if box["chunks"]:
                pending_chunks.extend(box["chunks"])
                box["chunks"] = []
            if box.get("step_dirty"):
                pending_steps.extend(box["steps"])
                box["steps"] = []
                box["step_dirty"] = False

        def _consumer() -> None:
            import asyncio

            async def _run() -> None:
                try:
                    astream = self.streaming_graph.astream(
                        {"messages": messages},
                        config={"recursion_limit": recursion_limit},
                        stream_mode=["updates", "messages"],
                    )
                    async for mode, data in astream:
                        if mode == "messages":
                            chunk, metadata = data
                            node = metadata.get("langgraph_node")
                            content = getattr(chunk, "content", None)
                            if node == "agent" and isinstance(content, str) and content:
                                box["chunks"].append(content)
                        else:  # updates
                            for node_name, node_output in data.items():
                                for msg in node_output.get("messages", []):
                                    if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                                        continue  # 最终答案的增量已全在 messages 分支推完
                                    recorded = self._record_message(node_name, msg)
                                    if recorded:
                                        box["steps"].append(recorded)
                                        box["step_dirty"] = True
                    box["done"] = True
                except Exception as exc:  # noqa: BLE001
                    box["error"] = exc

            try:
                asyncio.run(_run())
            except Exception as exc:  # noqa: BLE001
                # 当前线程已有运行中的事件循环等边界情况，记为流式不可用
                box["error"] = exc

        threading.Thread(target=_consumer, daemon=True).start()

        yielded_answer = False
        while True:
            _drain()
            for s in pending_steps:
                yield {"type": "step", "step": s}
                yielded_answer = False
            pending_steps.clear()
            for c in pending_chunks:
                yield {"type": "answer", "content": c}
                yielded_answer = True
            pending_chunks.clear()

            if box.get("error") is not None:
                logger.warning("流式 Agent 不可用，降级为非流式: %s", box["error"])
                self._streaming_known_broken = True
                # 已推过部分 token 时，只补整段答案会造成重复，故仅当没推过答案才补
                if not yielded_answer:
                    yield {"type": "streaming_fallback", "reason": "sse_unsupported"}
                    yield from self.run_stream(message, history)
                else:
                    yield {"type": "streaming_fallback", "reason": "sse_unsupported_mid_stream"}
                return

            if box.get("done"):
                break
            time.sleep(0.05)

        _drain()
        for s in pending_steps:
            yield {"type": "step", "step": s}
        pending_steps.clear()
        for c in pending_chunks:
            yield {"type": "answer", "content": c}
        pending_chunks.clear()

        if not yielded_answer:
            yield {"type": "answer", "content": "Agent 未能生成有效回答，请检查模型配置或稍后重试。"}

    def _build_messages(self, message: str, history: list[dict] | None) -> list:
        messages = []
        for item in (history or [])[-6:]:
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=message))
        return messages

    def _record_message(self, node_name: str, msg: Any) -> dict | None:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                calls = [
                    {"name": call["name"], "args": _jsonable(call.get("args", {}))}
                    for call in msg.tool_calls
                ]
                return {
                    "type": "tool_call",
                    "node": node_name,
                    "thought": msg.content or "选择工具执行任务",
                    "tool_calls": calls,
                }
            if msg.content:
                return {"type": "thought", "node": node_name, "content": msg.content}
        elif isinstance(msg, ToolMessage):
            return {
                "type": "observation",
                "node": node_name,
                "tool": getattr(msg, "name", None),
                "content": str(msg.content)[:2000],
            }
        return None

    @staticmethod
    def _is_recursion_error(exc: Exception) -> bool:
        return exc.__class__.__name__ in {"GraphRecursionError", "RecursionError"}


def _jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
