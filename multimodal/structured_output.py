"""结构化输出：任务 Prompt 模板 + Markdown 结果抽取与失败重试。"""
from __future__ import annotations

import logging
import re

from multimodal.vision_llm import VisionService

logger = logging.getLogger(__name__)

TASK_PROMPTS = {
    "describe": (
        "请详细描述这张图片的内容，包括其中的对象、场景、文字与模块之间的关系，"
        "使用简体中文，按要点组织，300 字以内。"
    ),
    "ocr": (
        "请对这张图片进行 OCR，完整、准确地提取其中所有文字。"
        "严格按以下格式输出，不要输出任何解释：\n"
        "```markdown\n<提取到的文字，保留原始换行与层级>\n```"
    ),
    "table": (
        "这是一张数据表/表格截图。请识别其中的表格结构，提取为 Markdown 表格。"
        "严格按以下格式输出，不要输出任何解释：\n"
        "```markdown\n| 列1 | 列2 |\n| --- | --- |\n| ... |\n```"
    ),
    "question": "请结合图片内容回答用户问题，若图片中没有依据请明确说明，使用简体中文。",
}

_FENCE_RE = re.compile(r"```(?:markdown|md|text)?\s*\n?(.*?)```", re.DOTALL)


def run_task(service: VisionService, data_url: str, task: str, question: str | None = None) -> dict:
    """执行指定的多模态任务，返回 {task, result, structured}。"""
    if task not in TASK_PROMPTS:
        raise ValueError(f"未知任务类型: {task}，支持 {list(TASK_PROMPTS)}")

    prompt = TASK_PROMPTS[task]
    if task == "question":
        if not question:
            raise ValueError("question 任务必须提供 question 参数")
        prompt = f"{prompt}\n用户问题：{question}"

    # OCR / 表格要求稳定结构化输出，抽取失败时重试一次
    structured_required = task in {"ocr", "table"}
    raw = service.analyze(data_url, prompt, retries=1 if structured_required else 0)
    result = extract_markdown(raw) if structured_required else raw.strip()

    if structured_required and not result:
        logger.info("结构化输出为空，重试一次")
        raw = service.analyze(data_url, prompt)
        result = extract_markdown(raw)

    return {"task": task, "result": result or raw.strip(), "raw": raw}


def extract_markdown(text: str) -> str:
    """从模型输出中抽取围栏代码块内容；无围栏时返回清洗后的原文。"""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    # 模型偶尔直接输出表格，直接保留
    return stripped
