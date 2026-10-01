"""工作流节点：input / llm / retrieval / tool / condition / output。

节点配置中的字符串支持模板变量：
    {{node_id}}      -> 上游节点的输出
    {{inputs.key}}   -> 运行时传入的输入参数
"""
from __future__ import annotations

import re
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

    def chat(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        client = get_openai_client()
        response = client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=0.3,
        )
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
    output = {k: inputs[k] for k in fields if k in inputs} if fields else dict(inputs)
    return output, {"fields": list(output)}


def _llm_node(config: dict, context: dict, services: NodeServices) -> tuple[Any, dict]:
    prompt = config.get("prompt")
    if not prompt:
        raise ValueError("llm 节点缺少 config.prompt")
    system = config.get(
        "system", "你是一个专业的 AI 工作流处理节点，请严格依据输入完成任务，使用简体中文。"
    )
    result = services.chat(prompt, system)
    return result, {"model": services.settings.llm_model, "chars": len(result)}


def _retrieval_node(config: dict, context: dict, services: NodeServices) -> tuple[Any, dict]:
    kb_id = config.get("kb_id")
    question = config.get("question")
    if not kb_id or not question:
        raise ValueError("retrieval 节点缺少 config.kb_id 或 config.question")
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
    if fields:
        output = {name: _resolve(ref, context) for name, ref in fields.items()}
    else:
        # 默认收集所有直接上游节点的输出
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
