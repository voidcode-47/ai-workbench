"""Embedding 封装：默认走 OpenAI 兼容接口；未配置密钥时降级为本地哈希向量。

本地哈希向量基于 hashing trick，仅用于无密钥环境下的离线演示与评测，
不具备真正的语义理解能力。
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import threading

from app.config import Settings
from app.exceptions import EmbeddingError
# 公开托管服务主机名单与 model_config 共用，避免两处漂移
from app.model_config import is_public_host

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
            # openai SDK ≥1.x 对空字符串直接抛 Missing credentials；
            # 本地/自建兼容端点不需要真 Key，用占位值（与 LLM 客户端一致）
            api_key=settings.openai_api_key or "sk-no-key-required",
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
_embedder_lock = threading.Lock()


def _use_remote_endpoint(settings: Settings) -> bool:
    """是否应调用远端 Embedding 接口。

    有 API Key 时走远端；没有 Key 但端点不是公开托管服务（如 LM Studio、
    vLLM、One-API 等本地/自建兼容网关）时同样走远端 —— 否则本地部署会被
    静默降级成语义无关的哈希向量，检索质量无从谈起。
    """
    if settings.openai_api_key:
        return True
    base = settings.embedding_base_url or settings.openai_base_url or ""
    if not base:
        return False
    return not is_public_host(base)


def _build_embedder(settings: Settings) -> BaseEmbedder:
    return OpenAIEmbedder(settings) if _use_remote_endpoint(settings) else LocalHashEmbedder()


def get_embedder(settings: Settings) -> BaseEmbedder:
    """单例 Embedder（pydantic Settings 不可哈希，不能直接用 lru_cache）。"""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = _build_embedder(settings)
    return _embedder


def reset_embedder() -> None:
    """丢弃单例，使下次调用按当前配置重建（切换 Embedding 模型/端点后调用）。"""
    global _embedder
    with _embedder_lock:
        _embedder = None


class EmbedderHandle(BaseEmbedder):
    """稳定句柄：每次调用都转发到当前生效的 Embedding 实现。

    RAGService 在构造时持有本句柄，因此切换 Embedding 模型后无需重建
    RAGService —— 重建会连带丢弃向量库的内存缓存。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def dim(self) -> int:
        return get_embedder(self._settings).dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return get_embedder(self._settings).embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return get_embedder(self._settings).embed_query(text)


def get_embedder_handle(settings: Settings) -> BaseEmbedder:
    """返回转发句柄（供长生命周期对象持有）。"""
    return EmbedderHandle(settings)
