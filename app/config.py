"""全局配置：通过 .env / 环境变量注入。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 应用
    app_name: str = "AI Workbench"
    api_key: str = ""
    # 初始管理员（首次启动自动创建；登录后请在个人设置中修改密码）
    admin_username: str = "admin"
    admin_password: str = "admin123"

    # LLM（OpenAI 兼容接口）
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    # Embedding 独立端点：留空则与 openai_base_url 一致（如本地 LM Studio 加载 bge-m3）
    embedding_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"
    vision_model: str = "gpt-4o"

    # RAG
    chunk_size: int = 800
    chunk_overlap: int = 120
    rag_top_k: int = 4

    # Agent
    agent_max_steps: int = 8
    agent_code_timeout: int = 10
    sqlite_path: str = "data/demo.db"
    # 是否启用 run_python 工具（任意代码执行，默认关闭；开启前请评估风险）
    enable_python_tool: bool = False

    # 服务
    data_dir: str = "data"
    max_upload_mb: int = 20
    enable_rate_limit: bool = True
    rate_limit_per_minute: int = 60
    # CORS 白名单（JSON 数组，如 ["http://localhost:8080"]）；留空 = 仅同源
    cors_origins: list[str] = Field(default_factory=list)
    # 工作流整体执行超时（秒），防止慢节点串行拖垮线程池
    workflow_timeout_seconds: int = 120

    @property
    def files_dir(self) -> Path:
        return Path(self.data_dir) / "files"

    @property
    def vector_db_dir(self) -> Path:
        return Path(self.data_dir) / "vector_db"

    @property
    def sqlite_full_path(self) -> Path:
        return Path(self.sqlite_path)

    def ensure_dirs(self) -> None:
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
