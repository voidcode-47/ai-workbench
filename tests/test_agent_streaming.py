"""Agent 逐 token 流式（打字机）与 SSE 网关降级的行为测试。

流式图 `build_agent_graph_streaming` 需要 LLM 端点支持 SSE。测试里无法真连
外网，故通过 monkeypatch `agent.graph.build_agent_graph_streaming` 返回的
compiled graph 的 `astream`，模拟"支持流式"和"不支持流式（抛错）"两种端点，
验证：
1. 流式成功 → 答案分块逐个产出
2. 流式失败（网关不支持）→ 自动降级到非流式路径，且标记
   `_streaming_known_broken`，后续调用不再重试流式
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.executor import AgentExecutor
from app.config import Settings


def _make_executor() -> AgentExecutor:
    return AgentExecutor(Settings())


def test_run_stream_typewriter_falls_back_on_sse_unsupported():
    """流式路径首字即抛错（模拟网关不支持 SSE）时，应降级为整段答案。"""
    ex = _make_executor()

    async def _fake_astream(*args, **kwargs):
        raise RuntimeError("gateway does not support SSE")
        yield  # pragma: no cover  # 让解释器识别为 async generator

    with (
        patch("agent.graph.build_agent_graph_streaming") as mock_build,
        patch.object(ex, "run_stream") as mock_run_stream,
    ):
        mock_build.return_value = MagicMock(astream=_fake_astream)
        mock_run_stream.return_value = iter(
            [
                {"type": "step", "step": {"type": "tool_call", "tool_calls": []}},
                {"type": "answer", "content": "这是最终答案"},
            ]
        )
        events = list(ex.run_stream_typewriter("hi"))

    assert events[0]["type"] == "streaming_fallback"
    assert "sse_unsupported" in events[0].get("reason", "")
    assert any(e["type"] == "answer" for e in events[1:])
    assert ex._streaming_known_broken is True


def test_run_stream_typewriter_skips_streaming_when_broken():
    """已确认流式不可用后，直接走非流式路径，不再尝试流式。"""
    ex = _make_executor()
    ex._streaming_known_broken = True

    with patch.object(ex, "run_stream") as mock_run_stream:
        mock_run_stream.return_value = iter([{"type": "answer", "content": "ok"}])
        events = list(ex.run_stream_typewriter("hi"))

    assert [e["type"] for e in events] == ["answer"]
    mock_run_stream.assert_called_once()


def test_typewriter_yields_answer_chunks_when_streaming_succeeds():
    """流式图正常产出 AIMessageChunk 时，应逐块转发 answer 事件。"""
    ex = _make_executor()

    from langchain_core.messages import AIMessageChunk

    async def _astream_ok(*args, **kwargs):
        for i in range(3):
            chunk = AIMessageChunk(content=f"tok{i}")
            yield "messages", (chunk, {"langgraph_node": "agent"})
        yield "updates", {"agent": {"messages": []}}

    with (
        patch("agent.graph.build_agent_graph_streaming") as mock_build,
        patch.object(ex, "_record_message", return_value=None),
    ):
        mock_build.return_value = MagicMock(astream=_astream_ok)
        events = list(ex.run_stream_typewriter("hi"))

    answer_events = [e for e in events if e["type"] == "answer"]
    assert len(answer_events) >= 3  # 至少 3 个 token 分块
    assert not any(e["type"] == "streaming_fallback" for e in events)
    assert ex._streaming_known_broken is False


def test_typewriter_no_answer_yields_placeholder():
    """流式成功但一个字都没产出的边缘情况，应补一个占位 answer。"""
    ex = _make_executor()

    async def _astream_empty(*args, **kwargs):
        yield "updates", {"agent": {"messages": []}}

    with (
        patch("agent.graph.build_agent_graph_streaming") as mock_build,
        patch.object(ex, "_record_message", return_value=None),
    ):
        mock_build.return_value = MagicMock(astream=_astream_empty)
        events = list(ex.run_stream_typewriter("hi"))

    assert any(e["type"] == "answer" for e in events)


def test_sse_error_event_masks_leaked_credentials(monkeypatch):
    """SSE error 事件必须与全局 HTTP 异常处理器走同一出口脱敏：
    上游异常文本里的 API Key（如 OpenAI 401 回显）不得原样发给浏览器。"""
    from fastapi.testclient import TestClient

    import api.agent_routes as routes
    import main
    from app.dependencies import require_user

    leaked_key = "sk-leakedsecret123456"

    class _BoomExecutor:
        def run_stream_typewriter(self, *args, **kwargs):
            raise RuntimeError(f"401 Incorrect API key provided: {leaked_key}")
            yield  # pragma: no cover  # 标记为 generator，异常在首次迭代时抛出

    monkeypatch.setattr(routes, "get_agent_executor", lambda: _BoomExecutor())
    main.app.dependency_overrides[require_user] = lambda: {"username": "t", "role": "employee"}
    try:
        client = TestClient(main.app)
        resp = client.post("/api/v1/agent/chat/stream", json={"message": "hi"})
        body = resp.text
    finally:
        main.app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert "event: error" in body
    assert leaked_key not in body, "SSE error 事件回显了原始凭据"
    assert "***" in body
