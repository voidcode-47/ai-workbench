"""Agent 路由：工具列表、Agent 对话（支持 SSE 流式）。"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.schemas import AgentRequest, AgentResponse, AgentStep, ToolInfo
from app.dependencies import get_agent_executor, require_user
from app.exceptions import mask_secrets

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_user)])


@router.get("/tools", response_model=list[ToolInfo], tags=["Agent"])
def list_tools():
    return [ToolInfo(**item) for item in get_agent_executor().list_tools()]


@router.post("/chat", response_model=AgentResponse, tags=["Agent"])
def chat(payload: AgentRequest):
    executor = get_agent_executor()
    history = [m.model_dump() for m in payload.history] if payload.history else None
    result = executor.run(payload.message, history=history)
    return AgentResponse(answer=result["answer"], steps=[AgentStep(**s) for s in result["steps"]])


@router.post("/chat/stream", tags=["Agent"])
def chat_stream(payload: AgentRequest):
    """SSE 流式对话：逐 token 打字机效果；网关不支持 SSE 时自动降级为整段。

    事件类型：
    - step：工具调用 / 观察结果
    - answer：答案分块（流式）或整段（降级）
    - streaming_fallback：降级通知（reason 说明原因）
    - error：执行错误
    """
    executor = get_agent_executor()
    history = [m.model_dump() for m in payload.history] if payload.history else None

    def event_stream():
        try:
            for event in executor.run_stream_typewriter(payload.message, history=history):
                yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            # 原文进日志（便于排查），发给客户端的必须脱敏：
            # 上游（如 OpenAI 401）异常文本会回显提交的 API Key
            logger.warning("Agent 流式执行失败: %s", exc)
            body = json.dumps({"message": mask_secrets(str(exc))}, ensure_ascii=False)
            yield f"event: error\ndata: {body}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
