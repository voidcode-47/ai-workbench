"""Agent 执行器：运行状态图、记录思考/工具调用日志、限制最大步数、防死循环。"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import Settings
from app.exceptions import LLMError

logger = logging.getLogger(__name__)


class AgentExecutor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._graph = None

    @property
    def graph(self):
        if self._graph is None:
            from agent.graph import build_agent_graph

            self._graph = build_agent_graph(self.settings)
        return self._graph

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
        """流式执行 Agent，逐步产出事件 dict。

        - {"type": "step", "step": {...}}   工具调用 / 观察结果（每步完成即推送）
        - {"type": "answer", "content": str}  最终答案（按 LLM token 分块推送）
        """
        messages = self._build_messages(message, history)
        # 一次完整 ReAct 步包含 agent + tools 两个节点，recursion_limit 即步数硬上限
        recursion_limit = self.settings.agent_max_steps * 2 + 2
        final_answer = ""
        pending_tokens: list[str] = []

        def _flush_pending() -> str:
            nonlocal pending_tokens
            text = "".join(pending_tokens)
            pending_tokens = []
            return text

        try:
            stream = self.graph.stream(
                {"messages": messages},
                config={"recursion_limit": recursion_limit},
                stream_mode=["updates", "messages"],
            )
            for mode, data in stream:
                if mode == "messages":
                    # 逐 token 流：仅转发 agent 节点的文本（用于最终答案打字机效果）
                    chunk, metadata = data
                    if metadata.get("langgraph_node") == "agent":
                        content = getattr(chunk, "content", None)
                        if isinstance(content, str):
                            pending_tokens.append(content)
                else:  # updates：节点完成后的状态增量
                    for node_name, node_output in data.items():
                        for msg in node_output.get("messages", []):
                            recorded = self._record_message(node_name, msg)
                            if recorded:
                                if isinstance(msg, AIMessage) and msg.tool_calls:
                                    # 工具调用参数的 JSON 流不展示，直接丢弃
                                    pending_tokens = []
                                yield {"type": "step", "step": recorded}
                            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                                final_answer = msg.content
                                text = _flush_pending() or final_answer
                                if text:
                                    yield {"type": "answer", "content": text}
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
