"""Pydantic 请求 / 响应模型（含参数校验）。"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# kb_id 作为物理隔离命名空间并参与文件系统路径拼接，必须严格限制字符集
_KB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_kb_id(v: str) -> str:
    if not _KB_ID_RE.fullmatch(v):
        raise ValueError("kb_id 只能包含字母、数字、中划线与下划线（1-64 字符）")
    return v


# ---------- 通用 ----------
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ErrorResponse(BaseModel):
    error: dict


# ---------- 认证 / 用户 ----------
class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class LoginResponse(BaseModel):
    token: str
    username: str
    role: str


class UserInfo(BaseModel):
    username: str
    role: str
    created_at: str | None = None


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=2, max_length=32)
    password: str = Field(min_length=6, max_length=128)
    role: str = "employee"


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=6, max_length=128)


# ---------- RAG ----------
class KBCreateRequest(BaseModel):
    kb_id: str = Field(min_length=1, max_length=64, description="知识库标识，用作物理隔离命名空间")
    name: str | None = Field(default=None, max_length=128)

    _validate_kb_id = field_validator("kb_id")(validate_kb_id)


class KBResponse(BaseModel):
    kb_id: str
    name: str | None = None
    documents: int
    chunks: int
    owner: str | None = None
    role: str | None = None  # owner=可读写 / readonly=只读


class DocumentResponse(BaseModel):
    doc_id: str
    kb_id: str
    filename: str
    chunks: int
    created_at: str | None = None


class QueryRequest(BaseModel):
    kb_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    history: list[ChatMessage] | None = None

    _validate_kb_id = field_validator("kb_id")(validate_kb_id)


class Citation(BaseModel):
    index: int
    content: str
    score: float
    source: str | None = None
    page: int | None = None
    doc_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


# ---------- Agent ----------
class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] | None = None


class AgentStep(BaseModel):
    type: str
    node: str | None = None
    thought: str | None = None
    content: str | None = None
    tool: str | None = None
    tool_calls: list[dict] | None = None


class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStep]


class ToolInfo(BaseModel):
    name: str
    description: str


# ---------- 多模态 ----------
class MultimodalResponse(BaseModel):
    task: str
    result: str
    model: str
    image_size: int


# ---------- 工作流 ----------
class WorkflowRunRequest(BaseModel):
    definition: dict[str, Any]
    inputs: dict[str, Any] = Field(default_factory=dict)


class WorkflowLog(BaseModel):
    node: str
    type: str
    status: str
    elapsed_ms: int | None = None
    meta: dict | None = None
    error: str | None = None


class WorkflowRunResponse(BaseModel):
    status: str
    outputs: dict[str, Any]
    logs: list[WorkflowLog]
    context: dict[str, Any]


class TemplateSummary(BaseModel):
    key: str
    name: str
    description: str
