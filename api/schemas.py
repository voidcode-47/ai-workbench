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


# ---------- 模型接入 ----------
class ModelProfile(BaseModel):
    """提供商接入参数；字段为 None 表示「不修改」。"""

    base_url: str | None = Field(default=None, max_length=300)
    api_key: str | None = Field(default=None, max_length=200)
    llm_model: str | None = Field(default=None, max_length=120)
    embedding_model: str | None = Field(default=None, max_length=120)
    embedding_base_url: str | None = Field(default=None, max_length=300)
    vision_model: str | None = Field(default=None, max_length=120)

    @field_validator("base_url", "embedding_base_url")
    @classmethod
    def _validate_endpoint(cls, v: str | None) -> str | None:
        # 端点会被 openai SDK 直接使用，限制为 http(s)，避免 file:// 之类的协议
        if v is None:
            return None
        value = v.strip()
        if value and not value.startswith(("http://", "https://")):
            raise ValueError("端点必须以 http:// 或 https:// 开头")
        return value


class ModelConfigRequest(BaseModel):
    provider: str = Field(min_length=1, max_length=32)
    profile: ModelProfile = Field(default_factory=ModelProfile)


class ModelProbeRequest(BaseModel):
    """拉取模型列表 / 连接测试：用请求中的参数，不落库。"""

    provider: str = Field(min_length=1, max_length=32)
    profile: ModelProfile = Field(default_factory=ModelProfile)


class ModelProviderInfo(BaseModel):
    key: str
    label: str
    default_base_url: str
    default_model: str
    default_embedding_model: str
    default_vision_model: str
    models: list[str]
    local: bool
    supports_vision: bool
    requires_api_key: bool
    has_env_key: bool


class EffectiveModelInfo(BaseModel):
    provider: str
    provider_label: str
    base_url: str
    llm_model: str
    embedding_model: str
    embedding_base_url: str
    vision_model: str
    has_api_key: bool
    api_key_hint: str


class SavedProfileInfo(BaseModel):
    """已保存的接入参数（不含明文 Key）。"""

    base_url: str = ""
    has_api_key: bool = False
    llm_model: str = ""
    embedding_model: str = ""
    embedding_base_url: str = ""
    vision_model: str = ""


class ModelConfigResponse(BaseModel):
    catalog: list[ModelProviderInfo]
    effective: EffectiveModelInfo
    saved: dict[str, SavedProfileInfo]


class ModelListResponse(BaseModel):
    models: list[str] = Field(default_factory=list)
    source: str = "catalog"  # remote=服务端实时返回 / catalog=回退到建议列表
    error: str | None = None


class ModelTestResponse(BaseModel):
    ok: bool
    stage: str  # init | auth | reachable | model | chat
    detail: str
    models: list[str] = Field(default_factory=list)
