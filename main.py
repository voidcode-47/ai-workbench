"""AI Workbench 应用入口。

运行方式：
    uvicorn main:app --reload
启动后访问：
    http://127.0.0.1:8000/         前端页面
    http://127.0.0.1:8000/docs     OpenAPI 文档
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.main import api_router
from app.config import get_settings
from app.dependencies import InMemoryRateLimiter
from app.exceptions import register_exception_handlers
from app.logger import setup_logging

settings = get_settings()
setup_logging()
settings.ensure_dirs()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="企业知识库 RAG + 工具增强 Agent + 多模态理解 + 工作流编排",
)

# 跨域策略：仅当显式配置 CORS_ORIGINS 白名单时放开跨域；留空则仅允许同源访问
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,  # 白名单模式下不携带凭据，避免与任意来源组合
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ---------- 限流中间件（内存滑动窗口） ----------
rate_limiter = InMemoryRateLimiter()


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if settings.enable_rate_limit and request.url.path.startswith("/api/"):
        client = request.client.host if request.client else "anonymous"
        identity = request.headers.get("authorization") or client
        if not rate_limiter.allow(identity, settings.rate_limit_per_minute):
            return JSONResponse(
                status_code=429,
                content={"error": {"code": "RATE_LIMITED", "message": "请求过于频繁，请稍后再试"}},
            )
    return await call_next(request)


# ---------- 异常处理 / 路由 ----------
register_exception_handlers(app)

# 初始化用户系统与知识库元数据表（幂等）
from app.auth import init_auth_db  # noqa: E402

init_auth_db()

app.include_router(api_router, prefix="/api/v1")

# ---------- 前端静态页面 ----------
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    # 开发模式禁用静态文件缓存，避免前端改动后浏览器仍加载旧文件
    class NoCacheStaticFiles(StaticFiles):
        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)
            response.headers["Cache-Control"] = "no-store"
            return response

    app.mount("/static", NoCacheStaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


logger.info("%s 启动完成，API 前缀 /api/v1", settings.app_name)
