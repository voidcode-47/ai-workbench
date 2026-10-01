"""工作流执行器：校验 -> 执行 -> 汇总日志；并内置 4 个示例流程模板。"""
from __future__ import annotations

import logging
from typing import Any

from app.exceptions import WorkflowExecutionError
from workflow.engine import WorkflowEngine
from workflow.nodes import NodeServices, run_node

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    def __init__(self, services: NodeServices):
        self.services = services
        self.engine = WorkflowEngine()

    def validate(self, definition: dict) -> dict:
        return self.engine.validate(definition)

    def run(self, definition: dict, inputs: dict | None = None) -> dict[str, Any]:
        parsed = self.engine.validate(definition)

        def runner(node: dict, context: dict) -> tuple[Any, dict]:
            return run_node(node, context, self.services)

        try:
            result = self.engine.execute(
                parsed,
                inputs or {},
                runner,
                timeout=self.services.settings.workflow_timeout_seconds,
            )
        except WorkflowExecutionError:
            raise
        except Exception as exc:
            logger.exception("工作流执行失败")
            raise WorkflowExecutionError(f"工作流执行失败: {exc}") from exc

        return {
            "status": "success",
            "outputs": result["outputs"],
            "logs": result["logs"],
            "context": self._serialize_context(result["context"]),
        }

    @staticmethod
    def _serialize_context(context: dict) -> dict:
        """移除内部字段，输出可 JSON 序列化的节点输入输出。"""
        return {
            nid: {k: v for k, v in data.items() if k != "status"}
            for nid, data in context.items()
            if not nid.startswith("__") and isinstance(data, dict)
        }


# ---------- 内置示例流程模板 ----------
WORKFLOW_TEMPLATES: dict[str, dict] = {
    "document_qa": {
        "name": "文档问答流",
        "description": "从知识库检索资料后交给 LLM 生成带引用的回答",
        "definition": {
            "nodes": [
                {"id": "in", "type": "input"},
                {"id": "retrieve", "type": "retrieval",
                 "config": {"kb_id": "demo", "question": "{{inputs.question}}", "top_k": 4}},
                {"id": "answer", "type": "llm",
                 "config": {"prompt": "参考资料：\n{{retrieve}}\n\n请根据以上资料回答问题：{{inputs.question}}"}},
                {"id": "out", "type": "output", "config": {"fields": {"answer": "answer"}}},
            ],
            "edges": [
                {"source": "in", "target": "retrieve"},
                {"source": "retrieve", "target": "answer"},
                {"source": "answer", "target": "out"},
            ],
        },
    },
    "agent_analysis": {
        "name": "Agent 分析流",
        "description": "先联网搜索，再让 LLM 汇总分析",
        "definition": {
            "nodes": [
                {"id": "in", "type": "input"},
                {"id": "search", "type": "tool",
                 "config": {"name": "web_search", "args": {"query": "{{inputs.topic}}"}}},
                {"id": "analyze", "type": "llm",
                 "config": {"prompt": "搜索结果：\n{{search}}\n\n请围绕「{{inputs.topic}}」整理一份要点分析。"}},
                {"id": "out", "type": "output", "config": {"fields": {"analysis": "analyze", "raw_search": "search"}}},
            ],
            "edges": [
                {"source": "in", "target": "search"},
                {"source": "search", "target": "analyze"},
                {"source": "analyze", "target": "out"},
            ],
        },
    },
    "image_understanding": {
        "name": "图片理解流",
        "description": "输入图片的文字描述，输出结构化分析（图片可先经多模态接口转描述）",
        "definition": {
            "nodes": [
                {"id": "in", "type": "input"},
                {"id": "describe", "type": "llm",
                 "config": {"prompt": "以下是对一张图片的描述：\n{{inputs.description}}\n\n请输出结构化说明：1) 场景 2) 对象 3) 可能用途。"}},
                {"id": "out", "type": "output", "config": {"fields": {"structured_description": "describe"}}},
            ],
            "edges": [
                {"source": "in", "target": "describe"},
                {"source": "describe", "target": "out"},
            ],
        },
    },
    "report_generation": {
        "name": "报告生成流",
        "description": "先生成提纲，再根据提纲扩写成完整报告",
        "definition": {
            "nodes": [
                {"id": "in", "type": "input"},
                {"id": "outline", "type": "llm",
                 "config": {"prompt": "请为主题「{{inputs.topic}}」生成一份报告提纲，包含 3-5 个章节。"}},
                {"id": "report", "type": "llm",
                 "config": {"prompt": "请依据以下提纲扩写为一份完整报告，每章不少于 100 字：\n{{outline}}"}},
                {"id": "out", "type": "output",
                 "config": {"fields": {"outline": "outline", "report": "report"}}},
            ],
            "edges": [
                {"source": "in", "target": "outline"},
                {"source": "outline", "target": "report"},
                {"source": "report", "target": "out"},
            ],
        },
    },
}


def list_templates() -> list[dict]:
    return [{"key": key, "name": t["name"], "description": t["description"]} for key, t in WORKFLOW_TEMPLATES.items()]


def get_template(key: str) -> dict:
    if key not in WORKFLOW_TEMPLATES:
        raise KeyError(key)
    return WORKFLOW_TEMPLATES[key]
