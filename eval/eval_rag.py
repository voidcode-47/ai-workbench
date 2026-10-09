"""RAG 检索效果评测脚本。

指标：
- Hit Rate@K：期望文档是否出现在 Top-K 结果中
- MRR：期望文档首次出现位置的倒数
- Keyword Recall：期望关键词被召回片段覆盖的比例
- Context Precision@K：Top-K 中命中至少一个关键词的片段占比

用法：
    python -m eval.eval_rag --kb demo --cases cases.json --top-k 4

cases.json 格式：
    [
      {
        "question": "年假有多少天？",
        "expected_doc_id": "可选，对应上传后返回的 doc_id",
        "keywords": ["年假", "5 天", "工作满一年"]
      }
    ]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.dependencies import get_rag_service

# 未提供 cases 文件时使用的占位样例（请替换为你自己业务知识库的问题）
DEMO_CASES = [
    {"question": "示例问题：知识库主要讲了什么？", "keywords": ["示例"]},
]


def load_cases(path: str | None) -> list[dict]:
    if not path:
        return DEMO_CASES
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("cases 文件必须是 JSON 数组")
    return data


def evaluate(kb_id: str, cases: list[dict], top_k: int) -> dict:
    service = get_rag_service()

    total = len(cases)
    hits = 0
    reciprocal_ranks: list[float] = []
    keyword_recalls: list[float] = []
    precisions: list[float] = []

    details = []
    for case in cases:
        question = case["question"]
        expected_doc = case.get("expected_doc_id")
        keywords = case.get("keywords", [])
        records = service.retrieve(kb_id, question, top_k=top_k)

        # Hit / MRR
        rank = next(
            (i + 1 for i, r in enumerate(records) if r.metadata.get("doc_id") == expected_doc),
            None,
        ) if expected_doc else None
        if rank:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        elif expected_doc:
            reciprocal_ranks.append(0)

        # 关键词召回
        joined = "\n".join(r.text for r in records).lower()
        if keywords:
            hit_kw = sum(1 for kw in keywords if kw.lower() in joined)
            keyword_recalls.append(hit_kw / len(keywords))

            relevant = sum(1 for r in records if any(kw.lower() in r.text.lower() for kw in keywords))
            precisions.append(relevant / len(records) if records else 0)

        details.append(
            {
                "question": question,
                "rank": rank,
                "top_sources": [r.metadata.get("filename") for r in records],
                "scores": [round(r.score, 4) for r in records],
            }
        )

    # hit_rate 的分子只在有 expected_doc_id 的用例上累加，分母必须同口径；
    # 否则用例文件混用了有无 expected_doc_id 的两类用例时会系统性低估命中率
    evaluable = sum(1 for c in cases if c.get("expected_doc_id"))

    return {
        "kb_id": kb_id,
        "cases": total,
        "top_k": top_k,
        "hit_rate": round(hits / evaluable, 4) if evaluable else None,
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else None,
        "keyword_recall": round(sum(keyword_recalls) / len(keyword_recalls), 4) if keyword_recalls else None,
        "context_precision": round(sum(precisions) / len(precisions), 4) if precisions else None,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索效果评测")
    parser.add_argument("--kb", default="demo", help="知识库 ID")
    parser.add_argument("--cases", default=None, help="评测用例 JSON 文件路径")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    get_settings().ensure_dirs()
    cases = load_cases(args.cases)
    report = evaluate(args.kb, cases, args.top_k)

    print("=" * 56)
    print(f"知识库: {report['kb_id']} | 用例数: {report['cases']} | Top-K: {report['top_k']}")
    print("-" * 56)
    for name in ("hit_rate", "mrr", "keyword_recall", "context_precision"):
        value = report[name]
        if value is not None:
            print(f"{name:>18}: {value:.4f}")
    print("-" * 56)
    for i, item in enumerate(report["details"], 1):
        print(f"{i}. {item['question']}")
        print(f"   rank={item['rank']} sources={item['top_sources']} scores={item['scores']}")
    print("=" * 56)


if __name__ == "__main__":
    main()
