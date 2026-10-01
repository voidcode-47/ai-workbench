"""multimodal.structured_output 单元测试：围栏抽取与任务编排（VisionService 用假实现）。"""
from __future__ import annotations

import pytest

from multimodal.structured_output import extract_markdown, run_task


class FakeVisionService:
    """按预设序列返回模型输出，记录调用次数与 retries 参数。"""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def analyze(self, data_url: str, prompt: str, *, retries: int = 1) -> str:
        self.calls.append({"retries": retries, "prompt": prompt})
        return self.replies.pop(0)


class TestExtractMarkdown:
    def test_extracts_markdown_fence_content(self):
        raw = "说明文字\n```markdown\n第一行\n第二行\n```\n结尾"
        assert extract_markdown(raw) == "第一行\n第二行"

    def test_accepts_md_and_text_fence_tags(self):
        assert extract_markdown("```md\n表格内容\n```") == "表格内容"
        assert extract_markdown("```text\n纯文本\n```") == "纯文本"

    def test_accepts_fence_without_language_tag(self):
        assert extract_markdown("```\ncode\n```") == "code"

    def test_returns_stripped_raw_text_when_no_fence(self):
        assert extract_markdown("  模型直接输出的内容\n") == "模型直接输出的内容"


class TestRunTask:
    def test_describe_returns_stripped_raw_text(self):
        service = FakeVisionService(["  这是一张图片的描述  "])
        result = run_task(service, "data:image/png;base64,xxx", "describe")
        assert result["task"] == "describe"
        assert result["result"] == "这是一张图片的描述"
        # 非结构化任务不需要接口内部重试
        assert service.calls[0]["retries"] == 0

    def test_ocr_extracts_fenced_content(self):
        service = FakeVisionService(["```markdown\n提取的文字\n```"])
        result = run_task(service, "data:image/png;base64,xxx", "ocr")
        assert result["result"] == "提取的文字"

    def test_ocr_retries_once_when_first_output_is_empty(self):
        service = FakeVisionService([
            "   ",  # 首次抽取为空
            "```markdown\n第二次成功\n```",
        ])
        result = run_task(service, "data:image/png;base64,xxx", "ocr")
        assert result["result"] == "第二次成功"
        assert len(service.calls) == 2

    def test_question_task_requires_question_argument(self):
        service = FakeVisionService([])
        with pytest.raises(ValueError, match="question"):
            run_task(service, "data:image/png;base64,xxx", "question")

    def test_question_task_appends_question_to_prompt(self):
        service = FakeVisionService(["回答"])
        run_task(service, "data:image/png;base64,xxx", "question", question="图中有几个人？")
        assert "图中有几个人？" in service.calls[0]["prompt"]

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="未知任务"):
            run_task(FakeVisionService([]), "data:image/png;base64,xxx", "summarize")
