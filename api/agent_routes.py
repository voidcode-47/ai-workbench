"""Agent 路由：工具列表、Agent 对话（支持 SSE 流式）。"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.schemas import AgentRequest, AgentResponse, AgentStep, ToolInfo
from app.dependencies import get_agent_executor, require_user

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
    """SSE 流式对话：逐步推送步骤事件与最终答案分块。"""
    executor = get_agent_executor()
    history = [m.model_dump() for m in payload.history] if payload.history else None

    def event_stream():
        try:
            for event in executor.run_stream(payload.message, history=history):
                yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            body = json.dumps({"message": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {body}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
