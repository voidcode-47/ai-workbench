"""图片预处理：魔数校验、大小限制、Base64 Data URL 编码。"""
from __future__ import annotations

import base64
import binascii

from app.config import Settings
from app.exceptions import AppError

# 支持的图片类型（魔数 -> mime）
_MAGIC_BYTES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "image/webp",  # webp: RIFF????WEBP
}


class ImageValidationError(AppError):
    status_code = 400
    code = "IMAGE_VALIDATION_ERROR"


def encode_image(raw: bytes, settings: Settings) -> dict:
    """校验并把图片编码为 data URL，返回 {mime, base64, data_url, size}。"""
    if not raw:
        raise ImageValidationError("图片内容为空")
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise ImageValidationError(f"图片过大，最大允许 {settings.max_upload_mb}MB")

    mime = _detect_mime(raw)
    if mime is None:
        raise ImageValidationError("不支持的图片格式，仅支持 PNG/JPEG/GIF/WEBP")
    if mime == "image/webp" and raw[8:12] != b"WEBP":
        raise ImageValidationError("非法的 WEBP 文件")

    encoded = base64.b64encode(raw).decode("ascii")
    try:
        base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ImageValidationError("图片 Base64 编码失败") from exc

    return {
        "mime": mime,
        "base64": encoded,
        "data_url": f"data:{mime};base64,{encoded}",
        "size": len(raw),
    }


def _detect_mime(raw: bytes) -> str | None:
    for magic, mime in _MAGIC_BYTES.items():
        if raw.startswith(magic):
            return mime
    return None
