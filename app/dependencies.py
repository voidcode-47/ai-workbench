"""公共依赖：鉴权、限流器、OpenAI 客户端、各模块服务单例。"""
from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict, deque
from functools import lru_cache

from fastapi import Depends, Header

from app.config import get_settings
from app.exceptions import AuthenticationError, ForbiddenError


# ---------- 用户鉴权（Bearer token） ----------
async def require_user(authorization: str | None = Header(default=None)) -> dict:
    """从 Authorization: Bearer <token> 解析当前用户，返回 {"username","role"}。"""
    from app.auth import verify_token

    if not authorization or not authorization.startswith("Bearer "):
        raise AuthenticationError("请先登录")
    user = verify_token(authorization[7:].strip())
    if user is None:
        raise AuthenticationError("登录已过期或无效，请重新登录")
    return user


async def require_admin(user: dict = Depends(require_user)) -> dict:
    if user["role"] != "admin":
        raise ForbiddenError("需要管理员权限")
    return user


# ---------- API Key 鉴权（旧接口兼容，未启用时不做校验） ----------
async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if settings.api_key and not secrets.compare_digest(x_api_key or "", settings.api_key):
        raise AuthenticationError("无效或缺失的 X-API-Key")


# ---------- 滑动窗口限流器 ----------
# 桶数超过阈值时清理过期桶，防止恶意伪造 x-api-key 导致内存无限增长
_MAX_BUCKETS = 4096


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, identity: str, limit: int, window_seconds: int = 60) -> bool:
        now = time.time()
        with self._lock:
            if len(self._hits) > _MAX_BUCKETS:
                # 优先驱逐请求数<=1的桶（伪造 identity 通常只发一次请求就换新 key）
                single_use = [k for k, v in self._hits.items() if len(v) <= 1]
                for key in single_use:
                    del self._hits[key]
                # 仍超限时按最旧活动时间驱逐
                if len(self._hits) > _MAX_BUCKETS:
                    ordered = sorted(
                        self._hits.items(),
                        key=lambda kv: kv[1][-1] if kv[1] else 0.0,
                    )
                    for key, _ in ordered[: len(self._hits) - _MAX_BUCKETS]:
                        del self._hits[key]
            bucket = self._hits[identity]
            while bucket and now - bucket[0] > window_seconds:
                bucket.popleft()
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True


# ---------- OpenAI 兼容客户端 ----------
@lru_cache(maxsize=4)
def get_openai_client():
    """返回 OpenAI 兼容 SDK 客户端（base_url 可指向任意兼容服务）。"""
    from openai import OpenAI  # 延迟导入，降低模块加载成本

    settings = get_settings()
    return OpenAI(
        api_key=settings.openai_api_key or "sk-no-key-required",
        base_url=settings.openai_base_url,
        timeout=60,
    )


# ---------- RAG 服务 ----------
@lru_cache
def get_rag_service():
    from rag.chat_chain import RAGChatChain
    from rag.embeddings import get_embedder
    from rag.retriever import RAGService
    from rag.text_splitter import TextSplitter
    from rag.vector_store import LocalVectorStore

    settings = get_settings()
    store = LocalVectorStore(settings.vector_db_dir)
    embedder = get_embedder(settings)
    splitter = TextSplitter(settings.chunk_size, settings.chunk_overlap)
    return RAGService(store=store, embedder=embedder, splitter=splitter, settings=settings)


@lru_cache
def get_chat_chain():
    from rag.chat_chain import RAGChatChain

    return RAGChatChain(retriever=get_rag_service(), settings=get_settings())


# ---------- Agent 服务 ----------
@lru_cache
def get_agent_executor():
    from agent.executor import AgentExecutor

    return AgentExecutor(settings=get_settings())


# ---------- 多模态服务 ----------
@lru_cache
def get_vision_service():
    from multimodal.vision_llm import VisionService

    return VisionService(settings=get_settings())


# ---------- 工作流服务 ----------
@lru_cache
def get_workflow_executor():
    from workflow.executor import WorkflowExecutor
    from workflow.nodes import NodeServices

    settings = get_settings()
    services = NodeServices(
        settings=settings,
        rag_service=get_rag_service(),
        agent_executor=get_agent_executor(),
    )
    return WorkflowExecutor(services=services)
