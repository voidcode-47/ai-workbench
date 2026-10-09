"""工作流节点：input / llm / retrieval / tool / condition / output。

节点配置中的字符串支持模板变量：
    {{node_id}}      -> 上游节点的输出
    {{inputs.key}}   -> 运行时传入的输入参数
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.dependencies import get_openai_client

NODE_TYPES = {"input", "llm", "retrieval", "tool", "condition", "output"}
_TEMPLATE_RE = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")

# 工作流允许调用的工具白名单：仅限无副作用或只读工具。
# run_python（任意代码执行）、read_file（读取服务器文件）不得通过工作流接口调用，
# 否则 POST /workflow/run 即成为公开代码执行入口。
WORKFLOW_SAFE_TOOLS = {"calculator", "current_date", "web_search", "query_database"}


@dataclass
class NodeServices:
    """节点执行所需的外部能力集合。"""

    settings: Settings
    rag_service: Any
    agent_executor: Any

    def chat(self, prompt: str, system: str | None = None, timeout: float | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        client = get_openai_client()
        # 显式传超时：否则一台卡住的端点会让单个节点远超工作流整体预算
        kwargs: dict = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": 0.3,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


def render(value: Any, context: dict) -> Any:
    """递归渲染字符串中的 {{占位符}}。"""
    if isinstance(value, str):
        def _replace(match: re.Match) -> str:
            return str(_resolve(match.group(1), context))
        return _TEMPLATE_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: render(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [render(v, context) for v in value]
    return value


def _resolve(path: str, context: dict) -> Any:
    if path.startswith("inputs."):
        key = path.split(".", 1)[1]
        return context.get("__inputs__", {}).get(key, "")
    parts = path.split(".")
    node_result = context.get(parts[0], {}).get("output", "")
    if len(parts) == 1:
        return node_result
    # 支持 {{node_id.key}} 访问字典输出
    for part in parts[1:]:
        if isinstance(node_result, dict):
            node_result = node_result.get(part, "")
        else:
            return ""
    return node_result


def run_node(node: dict, context: dict, services: NodeServices) -> tuple[Any, dict]:
    """执行单个节点，返回 (输出, 调试信息)。"""
    node_type = node["type"]
    config = render(node.get("config", {}), context)
    handler = _HANDLERS[node_type]
    return handler(config, context, services)


# ---------- 各类型节点处理函数 ----------
def _input_node(config: dict, context: dict, services: NodeServices) -> tuple[Any, dict]:
    fields = config.get("fields")
    inputs = context.get("__inputs__", {})
    if fields is None:
        # 未声明 fields = 暴露全部输入
        output = dict(inputs)
    else:
        # 显式空列表 = 不暴露任何输入（不能与"未声明"混为一谈）
        output = {k: inputs[k] for k in fields if k in inputs}
    return output, {"fields": list(output)}


def _remaining_timeout(context: dict) -> float | None:
    """按工作流整体截止时间算出本节点可用的剩余秒数。"""
    deadline = context.get("__deadline__")
    if not isinstance(deadline, (int, float)):
        return None
    return max(1.0, float(deadline) - time.time())


def _llm_node(config: dict, context: dict, services: NodeServices) -> tuple[Any, dict]:
    prompt = config.get("prompt")
    if not prompt:
        raise ValueError("llm 节点缺少 config.prompt")
    system = config.get(
        "system", "你是一个专业的 AI 工作流处理节点，请严格依据输入完成任务，使用简体中文。"
    )
    result = services.chat(prompt, system, timeout=_remaining_timeout(context))
    return result, {"model": services.settings.llm_model, "chars": len(result)}


def _retrieval_node(config: dict, context: dict, services: NodeServices) -> tuple[Any, dict]:
    kb_id = config.get("kb_id")
    question = config.get("question")
    if not kb_id or not question:
        raise ValueError("retrieval 节点缺少 config.kb_id 或 config.question")

    # 工作流定义由调用方完全掌控，这里不校验权限就等于绕过知识库 ACL：
    # 任何登录用户都能用一个 retrieval 节点读出他人私有库的内容
    user = context.get("__user__")
    if user is not None:
        from app.auth import kb_access_level
        from app.exceptions import ForbiddenError

        if kb_access_level(kb_id, user) == "none":
            raise ForbiddenError("知识库不存在或无权访问")

    records = services.rag_service.retrieve(kb_id, question, top_k=config.get("top_k"))
    if not records:
        return "", {"hits": 0}
    joined = "\n\n".join(
        f"[{i + 1}] {r.metadata.get('filename', '未知')}\n{r.text}"
        for i, r in enumerate(records)
    )
    return joined, {"hits": len(records), "kb_id": kb_id}


def _tool_node(config: dict, context: dict, services: NodeServices) -> tuple[Any, dict]:
    from agent.tools import get_tools

    tool_name = config.get("name")
    if not tool_name:
        raise ValueError("tool 节点缺少 config.name")
    if tool_name not in WORKFLOW_SAFE_TOOLS:
        raise ValueError(
            f"工作流仅允许调用安全工具: {sorted(WORKFLOW_SAFE_TOOLS)}"
        )
    tools = {t.name: t for t in get_tools(services.settings)}
    target = tools.get(tool_name)
    if target is None:
        raise ValueError(f"工具不存在: {tool_name}，可选: {sorted(tools)}")

    if "args" in config:
        result = target.invoke(config["args"])
    elif "input" in config:
        result = target.invoke(str(config["input"]))
    else:
        # 无参工具（如 current_date）
        result = target.invoke({})
    return str(result), {"tool": tool_name}


def _condition_node(config: dict, context: dict, services: NodeServices) -> tuple[Any, dict]:
    field = config.get("field")
    routes = config.get("routes", {})
    if not field or not routes:
        raise ValueError("condition 节点缺少 config.field 或 config.routes")
    value = _resolve(field, context)
    # routes: {分支标签: 匹配值}
    for label, expected in routes.items():
        if str(value) == str(expected):
            return label, {"value": value, "branch": label}
    default = config.get("default", "default")
    return default, {"value": value, "branch": default}


def _output_node(config: dict, context: dict, services: NodeServices) -> tuple[Any, dict]:
    fields = config.get("fields")
    if fields is not None:
        # 显式空 dict = 输出空结果（不能与"未声明"混为一谈）
        output = {name: _resolve(ref, context) for name, ref in fields.items()}
    else:
        # 未声明 fields：收集所有执行成功的节点输出
        output = {
            nid: data.get("output")
            for nid, data in context.items()
            if not nid.startswith("__") and isinstance(data, dict) and data.get("status") == "success"
        }
    return output, {"keys": list(output)}


_HANDLERS = {
    "input": _input_node,
    "llm": _llm_node,
    "retrieval": _retrieval_node,
    "tool": _tool_node,
    "condition": _condition_node,
    "output": _output_node,
}
