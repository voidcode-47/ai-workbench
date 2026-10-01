"""document_loader 单元测试：txt/md/csv 解析与错误处理（无需外部二进制依赖）。"""
from __future__ import annotations

import pytest

from app.exceptions import FileParseError
from rag.document_loader import load_file


class TestTextFiles:
    def test_load_txt_preserves_content_and_metadata(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("第一行\n第二行", encoding="utf-8")
        documents = load_file(path)
        assert len(documents) == 1
        assert documents[0].page_content == "第一行\n第二行"
        assert documents[0].metadata["filename"] == "notes.txt"
        assert documents[0].metadata["source"].endswith("notes.txt")

    def test_load_markdown_uses_text_loader(self, tmp_path):
        path = tmp_path / "doc.md"
        path.write_text("# 标题\n\n正文", encoding="utf-8")
        documents = load_file(path)
        assert documents[0].page_content == "# 标题\n\n正文"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileParseError, match="文件不存在"):
            load_file(tmp_path / "ghost.txt")

    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "a.exe"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(FileParseError, match="不支持的文件格式"):
            load_file(path)

    def test_empty_text_file_raises(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("   \n  ", encoding="utf-8")
        with pytest.raises(FileParseError, match="内容为空"):
            load_file(path)


class TestCsv:
    def test_csv_header_and_rows_joined_with_pipes(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("name,age\nAlice,30\nBob,25\n", encoding="utf-8")
        documents = load_file(path)
        assert documents[0].page_content == "name | age\nAlice | 30\nBob | 25"
        assert documents[0].metadata["filename"] == "data.csv"

    def test_csv_blank_rows_are_skipped(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("a,b\n1,2\n\n,,\n3,4\n", encoding="utf-8")
        documents = load_file(path)
        assert documents[0].page_content == "a | b\n1 | 2\n3 | 4"

    def test_csv_with_utf8_bom_is_handled(self, tmp_path):
        path = tmp_path / "bom.csv"
        path.write_text("\ufeff列1,列2\n值1,值2", encoding="utf-8")
        documents = load_file(path)
        assert documents[0].page_content == "列1 | 列2\n值1 | 值2"

    def test_empty_csv_raises(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(FileParseError, match="CSV 文件为空"):
            load_file(path)
