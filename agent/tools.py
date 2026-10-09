"""Agent 可调用的工具集合。

- calculator：基于 AST 的安全表达式计算（不使用 eval）
- current_date：当前日期/时间查询
- web_search：Bing RSS 实时检索（国内可直连，cn.bing.com 优先）
- query_database：只读 SQLite 查询（强制 SELECT、限制行数）
- read_file：读取上传目录中的文件（禁止路径穿越）
- run_python：子进程执行 Python 代码（带超时）
"""
from __future__ import annotations

import ast
import datetime as dt
import operator
import os
import re
import subprocess
import sys
from pathlib import Path

import httpx
from langchain_core.tools import tool

from app.config import Settings

# ---------- 计算器 ----------
_ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_CONSTANTS = (int, float)
# 整数幂指数上限：大指数会产生超大整数，耗尽 CPU/内存，
# 且结果转字符串还会触发 Python 4300 位整数转换限制
_MAX_POWER_EXPONENT = 1000
# 结果位宽上限：仅限制指数还不够——((10**1000)**1000)**1000 每一步指数都是 1000，
# 却会构造出约 10^9 位的整数（数百 MB）。按位宽预估可提前拦截。
_MAX_RESULT_BITS = 100_000


def _check_result_size(value: int) -> int:
    if value.bit_length() > _MAX_RESULT_BITS:
        raise ValueError(f"计算结果过大（超过 {_MAX_RESULT_BITS} 位），已拒绝")
    return value


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, _ALLOWED_CONSTANTS):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if type(node.op) is ast.Pow:
            if isinstance(right, int) and abs(right) > _MAX_POWER_EXPONENT:
                raise ValueError(f"幂运算指数绝对值不能超过 {_MAX_POWER_EXPONENT}")
            # 先按位宽预估，避免把超大整数真正算出来
            if isinstance(left, int) and isinstance(right, int) and right > 0:
                if left.bit_length() * right > _MAX_RESULT_BITS:
                    raise ValueError(
                        f"幂运算结果过大（预估超过 {_MAX_RESULT_BITS} 位），已拒绝"
                    )
        result = _ALLOWED_BINOPS[type(node.op)](left, right)
        return _check_result_size(result) if isinstance(result, int) else result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("仅支持数字与 + - * / // % ** 运算")


@tool
def calculator(expression: str) -> str:
    """安全计算器。输入一个数学表达式，例如 "(3 + 4) * 2 / 5"。"""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return str(result)
    except Exception as exc:
        return f"计算失败：{exc}"


@tool
def current_date() -> str:
    """获取当前的日期、星期与时间，用于回答'今天/现在'之类的问题。"""
    now = dt.datetime.now()
    weekdays = "一二三四五六日"
    return f"{now:%Y-%m-%d %H:%M:%S}，星期{weekdays[now.weekday()]}"


# ---------- 网页搜索 ----------
# 相同关键词短时间内的结果缓存：避免 Agent 循环重复搜索同一词（7B 模型常见行为）
import threading
import time

_search_cache: dict[str, tuple[float, str]] = {}
_search_cache_lock = threading.Lock()
_SEARCH_CACHE_TTL = 60.0  # 秒
_SEARCH_CACHE_MAX = 512   # 条数上限


def _cached_search(query: str) -> str | None:
    with _search_cache_lock:
        hit = _search_cache.get(query)
        if hit and time.time() - hit[0] < _SEARCH_CACHE_TTL:
            return hit[1]
    return None


def _store_search(query: str, result: str) -> None:
    now = time.time()
    with _search_cache_lock:
        # 缓存以 query 为 key 且只读时校验 TTL，不清理就会随不同关键词无限增长
        for key in [k for k, (ts, _) in _search_cache.items() if now - ts >= _SEARCH_CACHE_TTL]:
            _search_cache.pop(key, None)
        if len(_search_cache) >= _SEARCH_CACHE_MAX:
            oldest = sorted(_search_cache.items(), key=lambda kv: kv[1][0])
            for key, _ in oldest[: len(_search_cache) - _SEARCH_CACHE_MAX + 1]:
                _search_cache.pop(key, None)
        _search_cache[query] = (now, result)


@tool
def web_search(query: str) -> str:
    """联网搜索实时信息。输入搜索关键词，返回摘要结果（来自 Bing 搜索 RSS，国内可直连）。"""
    import xml.etree.ElementTree as ET

    cached = _cached_search(query)
    if cached is not None:
        return cached

    def _fetch(base: str) -> str | None:
        try:
            resp = httpx.get(
                base,
                params={"q": query, "format": "rss", "setlang": "zh-hans"},
                timeout=10,
                follow_redirects=True,
            )
            resp.raise_for_status()
            return resp.text
        except Exception:
            return None

    # 优先国内可直连的 cn.bing.com，失败自动降级到 www.bing.com
    xml_text = _fetch("https://cn.bing.com/search") or _fetch("https://www.bing.com/search")
    if not xml_text:
        result = "网页搜索失败：Bing 搜索连接超时，请稍后重试。"
        _store_search(query, result)
        return result

    snippets: list[str] = []
    try:
        root = ET.fromstring(xml_text)
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            desc = re.sub(r"<[^>]+>", "", item.findtext("description") or "").strip()
            if not title:
                continue
            text = f"{title}：{desc}" if desc else title
            snippets.append(f"{text}（来源：{link}）")
            if len(snippets) >= 6:
                break
    except Exception as exc:
        result = f"网页搜索解析失败：{exc}"
        _store_search(query, result)
        return result

    if not snippets:
        result = f"未搜索到与「{query}」相关的摘要结果。"
        _store_search(query, result)
        return result
    result = "\n".join(f"- {s}" for s in snippets)
    _store_search(query, result)
    return result


# ---------- 数据库查询 ----------
# 写操作 / 危险关键字：按词边界匹配（详见 query_database 内的说明）
_WRITE_KEYWORD_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|attach|detach|pragma|create|replace"
    r"|truncate|vacuum|reindex|grant|revoke)\b"
)
# 单次查询墙钟上限：只读模式挡不住递归 CTE 之类的自旋查询
_SQL_TIMEOUT_SECONDS = 5.0
_MAX_ROWS = 50


@tool
def query_database(sql: str) -> str:
    """对内置 SQLite 数据库执行只读 SELECT 查询，返回 Markdown 表格，最多 50 行。"""
    import sqlite3

    sql_stripped = sql.strip().rstrip(";")
    if not sql_stripped.lower().startswith(("select", "with")):
        return "拒绝执行：仅允许 SELECT / WITH 只读查询。"
    # 用词边界匹配而不是子串匹配：否则 updated_at / created_by / deleted_flag
    # 之类合法的列名会被误判为写操作而拒绝
    lowered = sql_stripped.lower()
    banned = _WRITE_KEYWORD_RE.search(lowered)
    if banned:
        return f"拒绝执行：查询中包含写操作或危险关键字（{banned.group(0)}）。"

    settings = _settings
    db_path = settings.sqlite_full_path
    if not db_path.exists():
        return f"数据库文件不存在：{db_path}（可在 {db_path} 放置一个 SQLite 文件）"
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            # 只读模式能挡住写操作，但挡不住 WITH RECURSIVE 这类长时间自旋的查询；
            # 用进度回调在超过时间预算后中断（SQLite 会抛 interrupted）
            deadline = time.monotonic() + _SQL_TIMEOUT_SECONDS

            def _abort_if_over_budget() -> int:
                return 1 if time.monotonic() > deadline else 0

            conn.set_progress_handler(_abort_if_over_budget, 10_000)
            cursor = conn.execute(sql_stripped)
            columns = [desc[0] for desc in cursor.description]
            # 多取一行用于判断"是否真的被截断"（正好 50 行时不应谎报截断）
            rows = cursor.fetchmany(_MAX_ROWS + 1)
        finally:
            conn.close()
    except Exception as exc:
        return f"数据库查询失败：{exc}"

    if not rows:
        return "查询成功，但没有返回数据。"

    truncated = len(rows) > _MAX_ROWS
    rows = rows[:_MAX_ROWS]
    table = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    table.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    suffix = f"\n\n（结果超过 {_MAX_ROWS} 行已截断）" if truncated else ""
    return "\n".join(table) + suffix


# ---------- 文件读取 ----------
@tool
def read_file(filename: str) -> str:
    """读取上传目录（data/files）中的文本文件内容，输入文件名或相对路径。"""
    settings = _settings
    base = settings.files_dir.resolve()
    try:
        target = (base / filename).resolve().relative_to(base)
    except ValueError:
        return "拒绝访问：路径越界。"
    target = base / target
    if not target.exists() or not target.is_file():
        return f"文件不存在：{filename}"
    if target.suffix.lower() not in {".txt", ".md", ".csv", ".json", ".log"}:
        return "仅允许读取文本类文件（txt/md/csv/json/log）。"
    try:
        return target.read_text(encoding="utf-8", errors="ignore")[:8000]
    except Exception as exc:
        return f"文件读取失败：{exc}"


# ---------- Python 代码执行 ----------
# 任意代码执行工具：默认不启用（见 get_tools），启用时也剔除敏感环境变量，
# 防止模型通过 print(os.environ) 泄露 API Key 等凭据。
_SENSITIVE_ENV_TOKENS = ("OPENAI", "API_KEY", "TOKEN", "SECRET", "PASSWORD")


def _clean_env() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if not any(tok in k.upper() for tok in _SENSITIVE_ENV_TOKENS)
    }


@tool
def run_python(code: str) -> str:
    """在子进程中执行 Python 代码并返回标准输出（有超时限制），适合数值计算与数据处理。"""
    settings = _settings
    try:
        env = _clean_env()
        env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=settings.agent_code_timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return f"执行超时（>{settings.agent_code_timeout}s），已终止。"
    except Exception as exc:
        return f"执行失败：{exc}"

    output = (completed.stdout or "") + (completed.stderr or "")
    output = output.strip() or "代码执行完成，无输出。"
    return output[:8000]


_settings: Settings = None  # type: ignore[assignment]


def get_tools(settings: Settings) -> list:
    """返回绑定了运行时配置的工具列表。

    run_python 为任意代码执行能力，默认不暴露给 Agent；
    确需使用时通过 ENABLE_PYTHON_TOOL=true 显式开启。
    """
    global _settings
    _settings = settings
    tools = [calculator, current_date, web_search, query_database, read_file]
    if settings.enable_python_tool:
        tools.append(run_python)
    return tools
