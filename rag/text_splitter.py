"""文本切分器：段落优先 + 超长段落滑动窗口，清洗特殊字符。"""
from __future__ import annotations

import re

from langchain_core.documents import Document

# 连续空白（含全角空格）压缩为单个空格
_WHITESPACE_RE = re.compile(r"[ \t\u3000]{2,}")
# 不可见控制字符
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(text: str) -> str:
    text = _CONTROL_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


class TextSplitter:
    """按字符长度切分，优先保留段落边界，长段落按 overlap 滑动切分。"""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            base_meta = dict(document.metadata)
            for piece in self._split_text(document.page_content):
                piece = clean_text(piece)
                if not piece:
                    continue
                metadata = dict(base_meta)
                metadata["chunk_index"] = len(chunks)
                chunks.append(Document(page_content=piece, metadata=metadata))
        return chunks

    def _split_text(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
        chunks: list[str] = []
        buffer = ""

        for paragraph in paragraphs:
            # 单段超长：先落盘已有缓冲，再对长段滑动切分
            if len(paragraph) > self.chunk_size:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                chunks.extend(self._sliding_window(paragraph))
                continue

            candidate = f"{buffer}\n{paragraph}" if buffer else paragraph
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = paragraph

        if buffer:
            chunks.append(buffer)
        return chunks

    def _sliding_window(self, text: str) -> list[str]:
        step = self.chunk_size - self.chunk_overlap
        chunks: list[str] = []
        for i in range(0, len(text), step):
            # 末窗完全落在前窗的重叠区内（没有任何新字符）时停止，
            # 否则高 overlap 下会产出一串纯重复尾块
            if i > 0 and len(text) - i <= self.chunk_overlap:
                break
            piece = text[i : i + self.chunk_size]
            if piece.strip():
                chunks.append(piece)
        return chunks
