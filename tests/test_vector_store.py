"""LocalVectorStore 单元测试：余弦检索、持久化、删除、维度校验与隔离。"""
from __future__ import annotations

import numpy as np
import pytest

from rag.vector_store import LocalVectorStore, VectorRecord
from app.exceptions import DimensionMismatchError


def _record(rid: str, doc_id: str, vector: list[float], text: str | None = None) -> VectorRecord:
    return VectorRecord(
        id=rid,
        text=text or rid,
        embedding=vector,
        metadata={"doc_id": doc_id, "filename": f"{doc_id}.txt"},
    )


@pytest.fixture
def store(tmp_path) -> LocalVectorStore:
    return LocalVectorStore(tmp_path / "vector_db")


class TestCollectionLifecycle:
    def test_create_list_exists_and_delete(self, store):
        assert store.list_collections() == []
        store.create_collection("kb1")
        store.create_collection("kb2")
        assert store.collection_exists("kb1")
        assert store.list_collections() == ["kb1", "kb2"]
        store.delete_collection("kb1")
        assert not store.collection_exists("kb1")
        assert store.list_collections() == ["kb2"]

    def test_create_collection_is_idempotent(self, store):
        store.create_collection("kb")
        store.create_collection("kb")
        assert store.list_collections() == ["kb"]


class TestAddAndSearch:
    def test_results_are_ranked_by_cosine_similarity(self, store):
        store.create_collection("kb")
        store.add("kb", [
            _record("a:0", "a", [1.0, 0.0]),
            _record("b:0", "b", [0.0, 1.0]),
            _record("c:0", "c", [0.9, 0.1]),
        ])
        results = store.search("kb", [1.0, 0.0], top_k=2)
        assert [r.id for r in results] == ["a:0", "c:0"]
        assert results[0].score == pytest.approx(1.0, abs=1e-6)

    def test_top_k_limits_result_count(self, store):
        store.create_collection("kb")
        store.add("kb", [_record(f"n{i}", "d", [1.0, float(i)]) for i in range(5)])
        assert len(store.search("kb", [1.0, 0.0], top_k=3)) == 3

    def test_search_empty_collection_returns_empty_list(self, store):
        store.create_collection("empty")
        assert store.search("empty", [1.0, 0.0]) == []

    def test_zero_query_vector_does_not_crash(self, store):
        store.create_collection("kb")
        store.add("kb", [_record("a:0", "a", [1.0, 0.0])])
        results = store.search("kb", [0.0, 0.0])
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.0, abs=1e-6)

    def test_embedding_dimension_mismatch_is_rejected(self, store):
        store.create_collection("kb")
        store.add("kb", [_record("a:0", "a", [1.0, 0.0])])
        with pytest.raises(DimensionMismatchError, match="维度"):
            store.add("kb", [_record("b:0", "b", [1.0, 0.0, 0.0])])

    def test_search_dimension_mismatch_gives_actionable_error(self, store):
        """换过 Embedding 模型却没重建索引时，检索应给出可操作提示而非 numpy 形状错误。"""
        store.create_collection("kb")
        store.add("kb", [_record("a:0", "a", [1.0, 0.0])])
        with pytest.raises(DimensionMismatchError, match="重建索引"):
            store.search("kb", [1.0, 0.0, 0.0], top_k=1)

    def test_search_with_doc_id_filter_scans_behind_top_ranks(self, store):
        # 目标文档的片段分数排名靠后（top_k 之外）时，
        # 按 doc_id 过滤仍必须能取到它，而不是先截断后过滤
        store.create_collection("kb")
        records = [_record(f"a:{i}", "a", [1.0, 0.0]) for i in range(4)]
        records.append(_record("b:0", "b", [0.9, 0.1]))
        store.add("kb", records)
        results = store.search("kb", [1.0, 0.0], top_k=4, doc_id="b")
        assert [r.id for r in results] == ["b:0"]
        assert all(r.metadata["doc_id"] == "b" for r in results)


class TestPersistence:
    def test_data_survives_store_reinstantiation(self, tmp_path):
        root = tmp_path / "vector_db"
        store = LocalVectorStore(root)
        store.create_collection("kb")
        store.add("kb", [_record("a:0", "a", [1.0, 0.0], text="持久化文本")])

        reloaded = LocalVectorStore(root)
        results = reloaded.search("kb", [1.0, 0.0])
        assert len(results) == 1
        assert results[0].text == "持久化文本"
        assert reloaded.list_documents("kb")[0]["chunks"] == 1

    def test_rebuild_replaces_entire_collection(self, store):
        store.create_collection("kb")
        store.add("kb", [
            _record("a:0", "a", [1.0, 0.0]),
            _record("a:1", "a", [0.0, 1.0]),
        ])
        store.rebuild("kb", [_record("b:0", "b", [1.0, 1.0])])
        docs = store.list_documents("kb")
        assert [d["doc_id"] for d in docs] == ["b"]
        assert docs[0]["chunks"] == 1

    def test_rebuild_with_empty_records_clears_collection(self, store):
        store.create_collection("kb")
        store.add("kb", [_record("a:0", "a", [1.0, 0.0])])
        store.rebuild("kb", [])
        assert store.search("kb", [1.0, 0.0]) == []
        assert store.list_documents("kb") == []
        # 清空后仍可重新写入
        store.add("kb", [_record("c:0", "c", [1.0, 0.0])])
        assert len(store.search("kb", [1.0, 0.0])) == 1


class TestDeleteDocument:
    def test_removes_only_matching_rows_and_reports_count(self, store):
        store.create_collection("kb")
        store.add("kb", [
            _record("a:0", "a", [1.0, 0.0]),
            _record("a:1", "a", [0.0, 1.0]),
            _record("b:0", "b", [1.0, 1.0]),
        ])
        assert store.delete_document("kb", "a") == 2
        assert store.delete_document("kb", "a") == 0
        results = store.search("kb", [1.0, 0.0], top_k=10)
        assert [r.id for r in results] == ["b:0"]

    def test_delete_from_empty_collection_returns_zero(self, store):
        store.create_collection("kb")
        assert store.delete_document("kb", "ghost") == 0


class TestCollectionsAreIsolated:
    def test_collections_do_not_share_records(self, store):
        store.create_collection("kb1")
        store.create_collection("kb2")
        store.add("kb1", [_record("a:0", "a", [1.0, 0.0])])
        assert store.search("kb2", [1.0, 0.0]) == []
        assert len(store.search("kb1", [1.0, 0.0])) == 1
