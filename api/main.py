"""API 路由聚合。"""
from __future__ import annotations

from fastapi import APIRouter

from api import agent_routes, auth_routes, multimodal_routes, rag_routes, workflow_routes

api_router = APIRouter()


@api_router.get("/health", tags=["System"])
def health():
    return {"status": "ok"}


api_router.include_router(auth_routes.router)
api_router.include_router(rag_routes.router, prefix="/rag")
api_router.include_router(agent_routes.router, prefix="/agent")
api_router.include_router(multimodal_routes.router, prefix="/multimodal")
api_router.include_router(workflow_routes.router, prefix="/workflow")
