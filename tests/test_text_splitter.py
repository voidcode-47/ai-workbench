"""文本切分器单元测试。"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.text_splitter import TextSplitter, clean_text


class TestSlidingWindow:
    def test_no_fully_duplicated_tail_when_overlap_large(self):
        # 12 字符、chunk=10、overlap=8：窗口 [2:12] 已覆盖全文末尾，
        # 起点 4/6/8/10 的切片全部被前一窗口覆盖，属于纯重复块，不应产出
        splitter = TextSplitter(chunk_size=10, chunk_overlap=8)
        pieces = splitter._sliding_window("0123456789AB")
        assert pieces == ["0123456789", "23456789AB"]

    def test_drops_tail_piece_fully_covered_by_previous_window(self):
        # 24 字符、chunk=10、overlap=3：[14:24] 已到文末，
        # [21:24] 完全落在前一窗口覆盖范围内
        splitter = TextSplitter(chunk_size=10, chunk_overlap=3)
        pieces = splitter._sliding_window("x" * 24)
        assert pieces == ["x" * 10, "x" * 10, "x" * 10]

    def test_keeps_short_tail_that_contains_new_content(self):
        # 25 字符时末窗 [21:25] 有 1 个新字符（超出重叠区），必须保留以覆盖全文
        splitter = TextSplitter(chunk_size=10, chunk_overlap=3)
        text = "abcdefghijklmnopqrstuvwxy"
        pieces = splitter._sliding_window(text)
        assert pieces[0] == text[:10]
        assert pieces[-1] == text[-len(pieces[-1]):]
        assert "".join(pieces) or pieces  # 块列表非空
        # 重组后文本首尾均被覆盖
        assert pieces[0].startswith("a")
        assert pieces[-1].endswith("y")

    def test_window_step_equals_chunk_size_minus_overlap(self):
        splitter = TextSplitter(chunk_size=10, chunk_overlap=0)
        text = "0123456789abcdefghij"  # 20 字符，无重叠
        pieces = splitter._sliding_window(text)
        assert pieces == ["0123456789", "abcdefghij"]


class TestSplitText:
    def test_short_text_returns_single_chunk(self):
        splitter = TextSplitter(chunk_size=800, chunk_overlap=120)
        assert splitter._split_text("一小段文本") == ["一小段文本"]

    def test_paragraphs_are_merged_up_to_chunk_size(self):
        splitter = TextSplitter(chunk_size=20, chunk_overlap=5)
        pieces = splitter._split_text("aaaa\nbbbb\ncccc")
        # 全部段落总长未超 20，合并为一个块
        assert pieces == ["aaaa\nbbbb\ncccc"]

    def test_long_paragraph_falls_back_to_sliding_window(self):
        splitter = TextSplitter(chunk_size=10, chunk_overlap=2)
        pieces = splitter._split_text("x" * 25)
        assert len(pieces) >= 3
        assert pieces[0] == "x" * 10

    def test_blank_paragraphs_are_ignored(self):
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        pieces = splitter._split_text("第一段\n\n\n   \n第二段")
        assert pieces == ["第一段\n第二段"]

    def test_overlap_must_be_smaller_than_chunk_size(self):
        with pytest.raises(ValueError):
            TextSplitter(chunk_size=10, chunk_overlap=10)
        with pytest.raises(ValueError):
            TextSplitter(chunk_size=10, chunk_overlap=11)


class TestSplitDocuments:
    def test_assigns_sequential_chunk_index_and_preserves_metadata(self):
        documents = [
            Document(page_content="内容一", metadata={"source": "a.txt"}),
            Document(page_content="内容二", metadata={"source": "b.txt"}),
        ]
        chunks = TextSplitter(chunk_size=800).split_documents(documents)
        assert len(chunks) == 2
        assert [c.metadata["chunk_index"] for c in chunks] == [0, 1]
        assert [c.metadata["source"] for c in chunks] == ["a.txt", "b.txt"]

    def test_whitespace_only_document_produces_no_chunks(self):
        chunks = TextSplitter(chunk_size=800).split_documents(
            [Document(page_content="\x00   \n  ", metadata={})]
        )
        assert chunks == []


class TestCleanText:
    def test_collapses_repeated_spaces_and_strips_control_characters(self):
        assert clean_text("a\x00  b\t\u3000\u3000c ") == "a b c"

    def test_strips_leading_and_trailing_whitespace(self):
        assert clean_text("  hello   world  ") == "hello world"

    def test_preserves_normal_newlines(self):
        assert clean_text("第一行\n第二行") == "第一行\n第二行"
