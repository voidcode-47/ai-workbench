"""多模态路由：图片上传 + 描述 / 问答 / OCR / 表格结构化。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from api.schemas import MultimodalResponse
from app.config import get_settings
from app.dependencies import get_vision_service, require_user
from app.exceptions import AppError
from multimodal.image_processor import encode_image
from multimodal.structured_output import TASK_PROMPTS, run_task

router = APIRouter(dependencies=[Depends(require_user)])


@router.post("/analyze", response_model=MultimodalResponse, tags=["Multimodal"])
async def analyze_image(
    file: UploadFile = File(...),
    task: str = Form(default="describe"),
    question: str | None = Form(default=None),
):
    if task not in TASK_PROMPTS:
        raise AppError(f"未知任务类型: {task}，支持 {list(TASK_PROMPTS)}")
    if task == "question" and not question:
        raise AppError("question 任务必须填写 question 字段")

    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if file.size is not None and file.size > max_bytes:
        raise AppError(f"图片过大，最大允许 {settings.max_upload_mb}MB")
    raw = await file.read()
    image = encode_image(raw, settings)
    result = run_task(get_vision_service(), image["data_url"], task, question)
    return MultimodalResponse(
        task=task,
        result=result["result"],
        model=settings.vision_model,
        image_size=image["size"],
    )
