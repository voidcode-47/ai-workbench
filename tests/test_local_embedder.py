"""LocalHashEmbedder 单元测试：确定性、L2 归一化、分词与相似度性质。"""
from __future__ import annotations

import math

from rag.embeddings import LocalHashEmbedder


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class TestLocalHashEmbedder:
    def setup_method(self):
        self.embedder = LocalHashEmbedder()

    def test_embedding_is_deterministic(self):
        assert self.embedder.embed_query("机器学习") == self.embedder.embed_query("机器学习")

    def test_embedding_dimension_is_512(self):
        assert len(self.embedder.embed_query("test")) == 512

    def test_embedding_is_l2_normalized(self):
        vector = self.embedder.embed_query("归一化测试 normalized test")
        norm = math.sqrt(sum(v * v for v in vector))
        assert math.isclose(norm, 1.0, rel_tol=1e-9)

    def test_tokenization_is_case_insensitive(self):
        assert self.embedder.embed_query("HELLO World") == self.embedder.embed_query("hello world")

    def test_shared_tokens_give_higher_similarity_than_disjoint_texts(self):
        anchor = self.embedder.embed_query("苹果")
        shared = _cosine(anchor, self.embedder.embed_query("苹果汁"))
        disjoint = _cosine(anchor, self.embedder.embed_query("香蕉葡萄"))
        # 中文按单字切分：「苹果」/「苹果汁」共享两个 token，「香蕉葡萄」无共享
        assert shared > 0.5
        assert shared > disjoint + 0.3

    def test_empty_text_returns_zero_vector_without_error(self):
        assert self.embedder.embed_query("") == [0.0] * 512

    def test_batch_embedding_matches_single(self):
        texts = ["第一句", "second text"]
        batch = self.embedder.embed_documents(texts)
        assert batch == [self.embedder.embed_query(t) for t in texts]
