"""视觉语言模型（VLM）调用：OpenAI 兼容的多模态 chat.completions 接口。"""
from __future__ import annotations

import logging
import time

from app.config import Settings
from app.dependencies import get_openai_client
from app.exceptions import LLMError

logger = logging.getLogger(__name__)


class VisionService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def analyze(self, data_url: str, prompt: str, *, retries: int = 1) -> str:
        """发送单张图片与文本指令，返回模型输出。"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        client = get_openai_client()
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.settings.vision_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=2000,
                )
                result = response.choices[0].message.content or ""
                logger.info("VLM 调用成功 model=%s length=%d", self.settings.vision_model, len(result))
                return result
            except Exception as exc:
                last_error = exc
                logger.warning("VLM 调用失败（第 %d 次）: %s", attempt + 1, exc)
                time.sleep(0.8 * (attempt + 1))
        raise LLMError(f"视觉模型调用失败: {last_error}") from last_error
