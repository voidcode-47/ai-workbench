"""Embedding 封装：默认走 OpenAI 兼容接口；未配置密钥时降级为本地哈希向量。

本地哈希向量基于 hashing trick，仅用于无密钥环境下的离线演示与评测，
不具备真正的语义理解能力。
"""
from __future__ import annotations

import hashlib
import logging
import math
import re

from app.config import Settings
from app.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

# 英文/数字按词，中日韩文字按单字切分（离线词级匹配对中文整串过于稀疏）
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_BATCH_SIZE = 16
_LOCAL_DIM = 512


class BaseEmbedder:
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, settings: Settings):
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.openai_api_key,
            # Embedding 支持独立端点（如本地 LM Studio），未配置时与 LLM 同端点
            base_url=settings.embedding_base_url or settings.openai_base_url,
            timeout=60,
        )
        self.model = settings.embedding_model
        self.dim = 1536  # 实际维度由模型决定，检索时以向量库中存储的维度为准

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        try:
            for start in range(0, len(texts), _BATCH_SIZE):
                batch = texts[start : start + _BATCH_SIZE]
                resp = self._client.embeddings.create(model=self.model, input=batch)
                results.extend(item.embedding for item in resp.data)
        except Exception as exc:
            raise EmbeddingError(f"Embedding 接口调用失败: {exc}") from exc
        self.dim = len(results[0])
        return results


class LocalHashEmbedder(BaseEmbedder):
    """无外部依赖的确定性向量（hashing trick + L2 归一化）。"""

    dim = _LOCAL_DIM

    def __init__(self) -> None:
        logger.warning("未配置 OPENAI_API_KEY，Embedding 降级为本地哈希向量（仅适合离线演示）")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


_embedder: BaseEmbedder | None = None


def get_embedder(settings: Settings) -> BaseEmbedder:
    """单例 Embedder（pydantic Settings 不可哈希，不能直接用 lru_cache）。"""
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbedder(settings) if settings.openai_api_key else LocalHashEmbedder()
    return _embedder
