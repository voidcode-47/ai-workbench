"""本地向量数据库：按知识库（collection）物理隔离，numpy 余弦相似度检索。

每个知识库落盘为一个目录：
    vector_db/<kb_id>/vectors.npy   向量矩阵
    vector_db/<kb_id>/meta.json    文本与元数据

线程安全：写路径（add/delete/rebuild/create/delete collection）由进程内锁保护，
落盘采用临时文件 + 原子替换，避免并发写互相覆盖或写一半损坏。
"""
from __future__ import annotations

import json
import shutil
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class VectorRecord:
    id: str
    text: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoredRecord:
    id: str
    text: str
    score: float
    metadata: dict


class LocalVectorStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # kb_id -> {"vectors": np.ndarray | None, "meta": list[dict]}
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()

    # ---------- 知识库管理 ----------
    def list_collections(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() if (p / "meta.json").exists())

    def collection_exists(self, kb_id: str) -> bool:
        return (self._dir(kb_id) / "meta.json").exists()

    def create_collection(self, kb_id: str) -> None:
        with self._lock:
            self._dir(kb_id).mkdir(parents=True, exist_ok=True)
            if not self.collection_exists(kb_id):
                self._persist(kb_id, np.zeros((0, 0)), [])
            self._cache.pop(kb_id, None)

    def delete_collection(self, kb_id: str) -> None:
        with self._lock:
            path = self._dir(kb_id)
            if path.exists():
                shutil.rmtree(path)
            self._cache.pop(kb_id, None)

    # ---------- 写入 / 删除 ----------
    def add(self, kb_id: str, records: list[VectorRecord]) -> None:
        if not records:
            return
        with self._lock:
            data = self._load(kb_id)
            new_vectors = np.asarray([r.embedding for r in records], dtype=np.float32)
            if data["vectors"] is None or data["vectors"].size == 0:
                vectors = new_vectors
            else:
                if data["vectors"].shape[1] != new_vectors.shape[1]:
                    raise ValueError("向量维度与库中已有数据不一致，可能更换了 Embedding 模型，请重建索引")
                vectors = np.vstack([data["vectors"], new_vectors])
            meta = data["meta"] + [
                {"id": r.id, "text": r.text, "metadata": r.metadata} for r in records
            ]
            self._persist(kb_id, vectors, meta)

    def delete_document(self, kb_id: str, doc_id: str) -> int:
        with self._lock:
            data = self._load(kb_id)
            if not data["meta"]:
                return 0
            kept = [m for m in data["meta"] if m["metadata"].get("doc_id") != doc_id]
            removed = len(data["meta"]) - len(kept)
            if removed == 0:
                return 0
            keep_indices = [i for i, m in enumerate(data["meta"]) if m["metadata"].get("doc_id") != doc_id]
            vectors = data["vectors"][keep_indices] if data["vectors"] is not None else np.zeros((0, 0))
            self._persist(kb_id, vectors, kept)
            return removed

    def rebuild(self, kb_id: str, records: list[VectorRecord]) -> None:
        """一次性原子替换整个知识库的索引（reindex 使用：全部准备成功后落盘）。"""
        with self._lock:
            if not records:
                self._persist(kb_id, np.zeros((0, 0)), [])
                return
            vectors = np.asarray([r.embedding for r in records], dtype=np.float32)
            meta = [
                {"id": r.id, "text": r.text, "metadata": r.metadata} for r in records
            ]
            self._persist(kb_id, vectors, meta)

    # ---------- 检索 ----------
    def search(
        self,
        kb_id: str,
        embedding: list[float],
        top_k: int = 4,
        doc_id: str | None = None,
    ) -> list[ScoredRecord]:
        data = self._load(kb_id)
        if not data["meta"] or data["vectors"] is None or len(data["vectors"]) == 0:
            return []

        query = np.asarray(embedding, dtype=np.float32)
        vectors = data["vectors"]
        q_norm = np.linalg.norm(query) or 1.0
        v_norm = np.linalg.norm(vectors, axis=1)
        v_norm[v_norm == 0] = 1.0
        scores = vectors @ query / (v_norm * q_norm)

        order = np.argsort(-scores)
        results: list[ScoredRecord] = []
        for idx in order:
            meta = data["meta"][idx]
            if doc_id and meta["metadata"].get("doc_id") != doc_id:
                continue
            results.append(
                ScoredRecord(
                    id=meta["id"],
                    text=meta["text"],
                    score=float(scores[idx]),
                    metadata=meta["metadata"],
                )
            )
            if len(results) >= top_k:
                break
        return results

    # ---------- 文档列表（按 doc_id 聚合） ----------
    def list_documents(self, kb_id: str) -> list[dict]:
        data = self._load(kb_id)
        grouped: dict[str, dict] = {}
        for item in data["meta"]:
            md = item["metadata"]
            doc_id = md.get("doc_id")
            if not doc_id:
                continue
            if doc_id not in grouped:
                grouped[doc_id] = {
                    "doc_id": doc_id,
                    "filename": md.get("filename", "unknown"),
                    "chunks": 0,
                    "stored_path": md.get("stored_path"),
                    "created_at": md.get("created_at"),
                }
            grouped[doc_id]["chunks"] += 1
        return sorted(grouped.values(), key=lambda x: x.get("created_at") or "")

    # ---------- 内部方法 ----------
    def _dir(self, kb_id: str) -> Path:
        return self.root / kb_id

    def _load(self, kb_id: str) -> dict:
        if kb_id in self._cache:
            return self._cache[kb_id]
        meta_path = self._dir(kb_id) / "meta.json"
        vectors_path = self._dir(kb_id) / "vectors.npy"
        if not meta_path.exists():
            data = {"vectors": np.zeros((0, 0)), "meta": []}
        else:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            vectors = np.load(vectors_path) if vectors_path.exists() and meta else np.zeros((0, 0))
            data = {"vectors": vectors, "meta": meta}
        self._cache[kb_id] = data
        return data

    def _persist(self, kb_id: str, vectors: np.ndarray, meta: list[dict]) -> None:
        directory = self._dir(kb_id)
        directory.mkdir(parents=True, exist_ok=True)
        # 临时文件 + 原子替换，避免写一半崩溃损坏索引。
        # np.save 要求文件名以 .npy 结尾，否则会自动追加后缀，故临时名形如 vectors.tmp.npy。
        tmp_vectors = directory / "vectors.tmp.npy"
        tmp_meta = directory / "meta.json.tmp"
        np.save(tmp_vectors, vectors)
        tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_meta.replace(directory / "meta.json")
        tmp_vectors.replace(directory / "vectors.npy")
        self._cache[kb_id] = {"vectors": vectors, "meta": meta}
