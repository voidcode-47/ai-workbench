"""检索服务门面：串联 加载 -> 切分 -> 嵌入 -> 向量库，按知识库隔离。"""
from __future__ import annotations

import datetime as dt
import re
import uuid
from pathlib import Path

from app.config import Settings
from app.exceptions import AppError
from rag.document_loader import load_file
from rag.embeddings import BaseEmbedder
from rag.text_splitter import TextSplitter
from rag.vector_store import LocalVectorStore, VectorRecord, ScoredRecord

# kb_id 参与文件系统路径拼接，服务层入口统一校验（与路由/schema 一致，纵深防御）
_KB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class InvalidKBId(AppError):
    status_code = 400
    code = "INVALID_KB_ID"


class KnowledgeBaseNotFound(AppError):
    status_code = 404
    code = "KB_NOT_FOUND"


class DocumentNotFound(AppError):
    status_code = 404
    code = "DOCUMENT_NOT_FOUND"


class RAGService:
    def __init__(
        self,
        store: LocalVectorStore,
        embedder: BaseEmbedder,
        splitter: TextSplitter,
        settings: Settings,
    ):
        self.store = store
        self.embedder = embedder
        self.splitter = splitter
        self.settings = settings

    @staticmethod
    def _validate_kb_id(kb_id: str) -> None:
        if not isinstance(kb_id, str) or not _KB_ID_RE.fullmatch(kb_id):
            raise InvalidKBId("kb_id 只能包含字母、数字、中划线与下划线（1-64 字符）")

    # ---------- 知识库 ----------
    def create_knowledge_base(self, kb_id: str) -> dict:
        self._validate_kb_id(kb_id)
        self.store.create_collection(kb_id)
        (self.settings.files_dir / kb_id).mkdir(parents=True, exist_ok=True)
        return {"kb_id": kb_id, "documents": 0}

    def list_knowledge_bases(self) -> list[dict]:
        result = []
        for kb_id in self.store.list_collections():
            docs = self.store.list_documents(kb_id)
            result.append(
                {"kb_id": kb_id, "documents": len(docs), "chunks": sum(d["chunks"] for d in docs)}
            )
        return result

    def delete_knowledge_base(self, kb_id: str) -> None:
        self._validate_kb_id(kb_id)
        self._ensure_kb(kb_id)
        self.store.delete_collection(kb_id)
        directory = self.settings.files_dir / kb_id
        if directory.exists():
            for file in directory.iterdir():
                file.unlink()
            directory.rmdir()

    def _ensure_kb(self, kb_id: str) -> None:
        if not self.store.collection_exists(kb_id):
            raise KnowledgeBaseNotFound(f"知识库不存在: {kb_id}")

    # ---------- 文档摄入 ----------
    def _prepare_records(
        self,
        file_path: str | Path,
        kb_id: str,
        doc_id: str,
        filename: str | None = None,
    ) -> tuple[list[VectorRecord], dict]:
        """加载 -> 切分 -> 嵌入，返回 (records, info)，不落库（reindex 复用）。

        filename 用于保留原始上传文件名（存储文件名可能带 doc_id 前缀）。
        """
        path = Path(file_path)
        display_name = filename or path.name
        created_at = dt.datetime.now().isoformat(timespec="seconds")

        documents = load_file(path)
        chunks = self.splitter.split_documents(documents)
        embeddings = self.embedder.embed_documents([c.page_content for c in chunks])

        records = [
            VectorRecord(
                id=f"{doc_id}:{i}",
                text=chunk.page_content,
                embedding=embeddings[i],
                metadata={
                    **chunk.metadata,
                    "doc_id": doc_id,
                    "filename": display_name,  # 覆盖为原始上传名（存储名可能带 doc_id 前缀）
                    "chunk_index": i,
                    "stored_path": str(path.resolve()),
                    "created_at": created_at,
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        info = {
            "doc_id": doc_id,
            "kb_id": kb_id,
            "filename": display_name,
            "chunks": len(records),
            "created_at": created_at,
        }
        return records, info

    def ingest_file(
        self,
        file_path: str | Path,
        kb_id: str,
        doc_id: str | None = None,
        filename: str | None = None,
    ) -> dict:
        self._validate_kb_id(kb_id)
        self._ensure_kb(kb_id)
        doc_id = doc_id or uuid.uuid4().hex
        records, info = self._prepare_records(file_path, kb_id, doc_id, filename)
        self.store.add(kb_id, records)
        return info

    def delete_document(self, kb_id: str, doc_id: str) -> None:
        self._validate_kb_id(kb_id)
        self._ensure_kb(kb_id)
        docs = {d["doc_id"]: d for d in self.store.list_documents(kb_id)}
        if doc_id not in docs:
            raise DocumentNotFound(f"文档不存在: {doc_id}")
        stored_path = docs[doc_id].get("stored_path")
        self.store.delete_document(kb_id, doc_id)
        if stored_path and Path(stored_path).exists():
            Path(stored_path).unlink()

    def list_documents(self, kb_id: str) -> list[dict]:
        self._validate_kb_id(kb_id)
        self._ensure_kb(kb_id)
        return self.store.list_documents(kb_id)

    def reindex(self, kb_id: str) -> dict:
        """依据原始上传文件重建向量索引（如更换切分参数/Embedding 模型后使用）。

        先全量重新加载并嵌入（失败不影响现有索引），全部成功后再一次性原子替换。
        """
        self._validate_kb_id(kb_id)
        self._ensure_kb(kb_id)
        documents = self.store.list_documents(kb_id)
        if not documents:
            return {"kb_id": kb_id, "reindexed": []}

        # 预检：所有源文件必须存在
        for doc in documents:
            stored_path = doc.get("stored_path")
            if not stored_path or not Path(stored_path).exists():
                raise AppError(f"文档原始文件已丢失，无法重建索引: {doc['filename']}")

        # 全量准备（不落库）：任一文档嵌入失败时，现有索引保持不变
        all_records: list[VectorRecord] = []
        for doc in documents:
            records, _ = self._prepare_records(
                doc["stored_path"], kb_id, doc["doc_id"], filename=doc.get("filename")
            )
            all_records.extend(records)

        # 全部成功后一次性原子替换
        self.store.rebuild(kb_id, all_records)
        return {
            "kb_id": kb_id,
            "reindexed": [
                {
                    "doc_id": doc["doc_id"],
                    "filename": doc["filename"],
                    "chunks": doc["chunks"],
                    "created_at": doc.get("created_at"),
                }
                for doc in documents
            ],
        }

    # ---------- 检索 ----------
    def retrieve(self, kb_id: str, question: str, top_k: int | None = None) -> list[ScoredRecord]:
        self._validate_kb_id(kb_id)
        self._ensure_kb(kb_id)
        top_k = top_k or self.settings.rag_top_k
        query_embedding = self.embedder.embed_query(question)
        return self.store.search(kb_id, query_embedding, top_k=top_k)
