"""统一业务异常与全局异常处理器。"""
from __future__ import annotations

import logging
import re

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# 上游 SDK 的异常文本可能回显提交的凭据（如 OpenAI 的 "Incorrect API key provided: sk-..."），
# 而 AppError 的 message 会原样返回给客户端，因此在出口处做脱敏。
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9._\-]{6,}"),
    re.compile(r"(?i)\b(api[_\-]?key|access[_\-]?token|secret)\b\s*[:=]?\s*\S{6,}"),
    re.compile(r"Bearer\s+\S{8,}"),
)


def mask_secrets(text: str) -> str:
    """抹掉文本中的疑似凭据，保留其余可读信息。"""
    masked = text
    for pattern in _SECRET_PATTERNS:
        masked = pattern.sub("***", masked)
    return masked


class AppError(Exception):
    """业务异常基类。"""

    status_code: int = 400
    code: str = "APP_ERROR"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AuthenticationError(AppError):
    status_code = 401
    code = "UNAUTHORIZED"


class ForbiddenError(AppError):
    status_code = 403
    code = "FORBIDDEN"


class RateLimitError(AppError):
    status_code = 429
    code = "RATE_LIMITED"


class FileParseError(AppError):
    status_code = 400
    code = "FILE_PARSE_ERROR"


class DimensionMismatchError(AppError):
    """向量维度与库中已有数据不一致（通常是换了 Embedding 模型却没重建索引）。"""

    status_code = 409
    code = "EMBEDDING_DIM_MISMATCH"


class EmbeddingError(AppError):
    status_code = 502
    code = "EMBEDDING_ERROR"


class LLMError(AppError):
    status_code = 502
    code = "LLM_ERROR"


class ToolExecutionError(AppError):
    status_code = 400
    code = "TOOL_ERROR"


class WorkflowValidationError(AppError):
    status_code = 400
    code = "WORKFLOW_VALIDATION_ERROR"


class WorkflowExecutionError(AppError):
    status_code = 500
    code = "WORKFLOW_EXECUTION_ERROR"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError):
        if exc.status_code >= 500:
            # 原文进日志（便于排查），返回给客户端的已脱敏
            logger.error("业务异常: %s | %s", exc.code, exc.message)
        payload = {"error": {"code": exc.code, "message": mask_secrets(exc.message)}}
        if exc.details:
            payload["error"]["details"] = exc.details
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception):
        logger.exception("未处理异常: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "INTERNAL_ERROR", "message": "服务内部错误"}},
        )
