"""RAG 问答链：检索片段 -> 组装带引用的 Prompt -> 调用 LLM -> 答案 + 引用溯源。"""
from __future__ import annotations

import logging
import time

from app.config import Settings
from app.dependencies import get_openai_client
from app.exceptions import LLMError
from rag.retriever import RAGService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是严谨的企业知识库问答助手。请遵守以下规则：
1. 只能依据【参考资料】回答问题，不得编造资料中不存在的事实；
2. 当参考资料不足以回答时，明确说明"根据现有资料无法回答该问题"；
3. 在回答中引用资料时使用 [1]、[2] 这样的编号标注来源；
4. 回答使用简体中文，条理清晰，必要时分点陈述。"""

QUESTION_TEMPLATE = """【参考资料】
{context}

【用户问题】
{question}

请基于以上参考资料回答，并在关键结论后标注引用编号。"""

EMPTY_CONTEXT_TEMPLATE = """【用户问题】
{question}

知识库中没有检索到任何相关资料。请直接告知用户：根据现有知识库资料无法回答该问题，并建议补充相关文档。"""


class RAGChatChain:
    def __init__(self, retriever: RAGService, settings: Settings):
        self.retriever = retriever
        self.settings = settings

    def answer(
        self,
        kb_id: str,
        question: str,
        history: list[dict] | None = None,
        top_k: int | None = None,
    ) -> dict:
        started = time.time()
        scored = self.retriever.retrieve(kb_id, question, top_k)

        citations = [
            {
                "index": i + 1,
                "content": record.text,
                "score": round(record.score, 4),
                "source": record.metadata.get("filename"),
                "page": record.metadata.get("page"),
                "doc_id": record.metadata.get("doc_id"),
            }
            for i, record in enumerate(scored)
        ]

        if scored:
            context = "\n\n".join(
                f"[{i + 1}] 来源：{r.metadata.get('filename', '未知')}"
                + (f" 第 {r.metadata['page']} 页" if r.metadata.get("page") else "")
                + f"\n{r.text}"
                for i, r in enumerate(scored)
            )
            user_content = QUESTION_TEMPLATE.format(context=context, question=question)
        else:
            user_content = EMPTY_CONTEXT_TEMPLATE.format(question=question)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-6:])  # 只携带最近 3 轮，控制 token
        messages.append({"role": "user", "content": user_content})

        answer_text = self._chat(messages)
        logger.info("RAG 问答完成 kb=%s 片段=%d 耗时=%.2fs", kb_id, len(scored), time.time() - started)
        return {"answer": answer_text, "citations": citations}

    def _chat(self, messages: list[dict], *, retries: int = 2) -> str:
        client = get_openai_client()
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=messages,
                    temperature=0.3,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:  # 网络/限流/服务端错误统一重试
                last_error = exc
                logger.warning("LLM 调用失败（第 %d 次）: %s", attempt + 1, exc)
                time.sleep(0.8 * (attempt + 1))
        raise LLMError(f"大模型调用失败: {last_error}") from last_error
