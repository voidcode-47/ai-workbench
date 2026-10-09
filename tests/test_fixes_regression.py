"""缺陷修复回归测试。

覆盖本轮修复的问题点，防止回归：
- 知识库所有权可被他人抢占（create 时改写 owner）
- 工作流 retrieval 节点绕过知识库 ACL
- 限流 key 使用可伪造的 Authorization 头
- 改密后旧 token 仍然有效（require_user 回查 token_version）
- calculator 嵌套幂构造超大整数
- 网页搜索缓存无上限增长
- query_database 在恰好 50 行时谎报截断
- 终止答案被当作 thought 步骤重复推送
- 上游异常文本回显 API Key
- eval hit_rate 分母口径错误
- 空 fields 被当成"未声明"
- 工作流节点配置错误返回 500 而非 400
- Embedding 空 Key 崩溃 / 公开端点无 Key 未降级
"""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from agent.executor import AgentExecutor
from agent.tools import calculator
from app.exceptions import (
    DimensionMismatchError,
    ForbiddenError,
    WorkflowValidationError,
    mask_secrets,
)
from api.schemas import KBCreateRequest
from workflow.nodes import _input_node, _output_node, _retrieval_node


# ============================================================
# 知识库权限
# ============================================================
class TestKbAccessLevel:
    def test_matrix(self, monkeypatch):
        import app.auth as auth_mod

        monkeypatch.setattr(auth_mod, "kb_get_meta", lambda kb_id: {"owner": "admin"})
        assert auth_mod.kb_access_level("kb", {"username": "admin", "role": "admin"}) == "owner"
        assert auth_mod.kb_access_level("kb", {"username": "bob", "role": "employee"}) == "readonly"

        monkeypatch.setattr(auth_mod, "kb_get_meta", lambda kb_id: {"owner": "alice"})
        assert auth_mod.kb_access_level("kb", {"username": "alice", "role": "employee"}) == "owner"
        assert auth_mod.kb_access_level("kb", {"username": "bob", "role": "employee"}) == "none"
        # 管理员也看不到员工的私有库
        assert auth_mod.kb_access_level("kb", {"username": "root", "role": "admin"}) == "none"

        # 历史知识库（无元数据）视为 admin 系统库
        monkeypatch.setattr(auth_mod, "kb_get_meta", lambda kb_id: None)
        assert auth_mod.kb_access_level("kb", {"username": "bob", "role": "employee"}) == "readonly"


class TestCreateKnowledgeBaseOwnership:
    """create 使用 INSERT OR REPLACE 写 owner，必须先校验归属，否则可被抢占。"""

    def _patch(self, monkeypatch, existing_meta, known_ids):
        import api.rag_routes as routes

        monkeypatch.setattr(routes, "kb_get_meta", lambda kb_id: existing_meta)
        monkeypatch.setattr(
            routes,
            "get_rag_service",
            lambda: SimpleNamespace(
                list_knowledge_bases=lambda: [{"kb_id": k} for k in known_ids],
                create_knowledge_base=lambda kb_id: None,
            ),
        )
        written = {}
        monkeypatch.setattr(routes, "kb_set_meta", lambda kb_id, name, owner: written.update(
            {"kb_id": kb_id, "name": name, "owner": owner}
        ))
        return written

    def test_cannot_take_over_foreign_kb(self, monkeypatch):
        written = self._patch(monkeypatch, {"owner": "admin", "name": "公共库"}, ["demo"])
        from api.rag_routes import create_knowledge_base

        with pytest.raises(ForbiddenError):
            create_knowledge_base(
                KBCreateRequest(kb_id="demo", name="mine"),
                user={"username": "bob", "role": "employee"},
            )

        assert written == {}, "越权请求不得写入 kb_meta"

    def test_legacy_kb_without_meta_is_admin_owned(self, monkeypatch):
        # 只在向量库里有、没有元数据的旧库按 admin 系统库处理，普通员工不得认领
        self._patch(monkeypatch, None, ["legacy"])
        from api.rag_routes import create_knowledge_base

        with pytest.raises(ForbiddenError):
            create_knowledge_base(
                KBCreateRequest(kb_id="legacy", name="mine"),
                user={"username": "bob", "role": "employee"},
            )

    def test_owner_can_recreate_own_kb_idempotently(self, monkeypatch):
        written = self._patch(monkeypatch, {"owner": "bob", "name": "旧名"}, ["mine"])
        from api.rag_routes import create_knowledge_base

        resp = create_knowledge_base(
            KBCreateRequest(kb_id="mine", name="新名"),
            user={"username": "bob", "role": "employee"},
        )

        assert resp.owner == "bob"
        assert written["owner"] == "bob"

    def test_new_kb_is_created_for_caller(self, monkeypatch):
        written = self._patch(monkeypatch, None, [])
        from api.rag_routes import create_knowledge_base

        resp = create_knowledge_base(
            KBCreateRequest(kb_id="fresh", name="n"),
            user={"username": "bob", "role": "employee"},
        )

        assert resp.owner == "bob"
        assert written == {"kb_id": "fresh", "name": "n", "owner": "bob"}


# ============================================================
# 工作流 retrieval 节点 ACL
# ============================================================
class TestWorkflowRetrievalAcl:
    def _services(self):
        return SimpleNamespace(
            settings=SimpleNamespace(llm_model="m"),
            rag_service=SimpleNamespace(retrieve=lambda *a, **k: []),
        )

    def test_denies_kb_the_user_cannot_see(self, monkeypatch):
        import app.auth as auth_mod

        monkeypatch.setattr(auth_mod, "kb_access_level", lambda kb_id, user: "none")

        with pytest.raises(ForbiddenError):
            _retrieval_node(
                {"kb_id": "hr", "question": "薪资结构"},
                {"__inputs__": {}, "__user__": {"username": "bob", "role": "employee"}},
                self._services(),
            )

    def test_allows_readonly_kb(self, monkeypatch):
        import app.auth as auth_mod

        monkeypatch.setattr(auth_mod, "kb_access_level", lambda kb_id, user: "readonly")
        captured = {}

        def _retrieve(kb_id, question, top_k=None):
            captured["kb_id"] = kb_id
            return []

        services = SimpleNamespace(
            settings=SimpleNamespace(llm_model="m"),
            rag_service=SimpleNamespace(retrieve=_retrieve),
        )

        output, meta = _retrieval_node(
            {"kb_id": "pub", "question": "q"},
            {"__inputs__": {}, "__user__": {"username": "bob", "role": "employee"}},
            services,
        )

        assert captured["kb_id"] == "pub"
        assert output == ""
        assert meta["hits"] == 0


# ============================================================
# 限流
# ============================================================
def test_rate_limit_not_bypassable_with_forged_authorization(monkeypatch):
    """旧实现用 Authorization 头做限流 key，每次换一个伪造值就能拿到全新桶。"""
    from fastapi.testclient import TestClient

    import main

    monkeypatch.setattr(main.settings, "enable_rate_limit", True)
    monkeypatch.setattr(main.settings, "rate_limit_per_minute", 60)
    main.rate_limiter._hits.clear()

    client = TestClient(main.app)
    codes = [
        client.post(
            "/api/v1/auth/login",
            json={"username": "ghost", "password": "nope"},
            headers={"Authorization": f"Bearer forged-{i}"},
        ).status_code
        for i in range(14)
    ]

    assert 429 in codes, f"伪造 Authorization 也应被限流，实际状态码: {codes}"


def test_login_limiter_uses_a_separate_bucket(monkeypatch):
    """登录与常规接口不能共用一个计数器，否则常态调用会把登录配额挤没。"""
    from fastapi.testclient import TestClient

    import main

    monkeypatch.setattr(main.settings, "enable_rate_limit", True)
    monkeypatch.setattr(main.settings, "rate_limit_per_minute", 60)
    main.rate_limiter._hits.clear()

    client = TestClient(main.app)
    # 用常规接口消耗远多于登录限额（10）的请求数
    for _ in range(12):
        client.get("/api/v1/health")

    resp = client.post("/api/v1/auth/login", json={"username": "ghost", "password": "x"})

    assert resp.status_code == 401, f"登录不应被常规接口请求挤掉，实际 {resp.status_code}"


# ============================================================
# 工具
# ============================================================
class TestCalculatorGuards:
    def test_nested_power_is_rejected(self):
        # 每一步指数都是 1000，旧的指数检查放行，却会构造约 10^9 位的整数
        out = calculator.invoke({"expression": "((10**1000)**1000)**1000"})
        assert "计算失败" in out
        assert "过大" in out

    def test_large_exponent_still_rejected(self):
        assert "计算失败" in calculator.invoke({"expression": "2 ** 200000"})

    def test_normal_power_still_works(self):
        assert calculator.invoke({"expression": "2 ** 10"}) == "1024"


def test_search_cache_is_bounded(monkeypatch):
    import agent.tools as tools

    monkeypatch.setattr(tools, "_SEARCH_CACHE_MAX", 5)
    tools._search_cache.clear()

    for i in range(50):
        tools._store_search(f"q{i}", "r")

    assert len(tools._search_cache) <= 5


class TestQueryDatabaseTruncation:
    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        import agent.tools as tools

        path = tmp_path / "demo.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(51)])
        conn.commit()
        conn.close()
        monkeypatch.setattr(tools, "_settings", SimpleNamespace(sqlite_full_path=path))
        return tools

    def test_no_false_truncation_at_exactly_50_rows(self, db):
        out = db.query_database.invoke({"sql": "SELECT * FROM t LIMIT 50"})
        assert "已截断" not in out

    def test_reports_truncation_when_more_than_50(self, db):
        out = db.query_database.invoke({"sql": "SELECT * FROM t"})
        assert "已截断" in out


# ============================================================
# Agent 事件流
# ============================================================
class _FakeGraph:
    def __init__(self, updates):
        self._updates = updates

    def stream(self, *args, **kwargs):
        return iter(self._updates)


def _executor(updates) -> AgentExecutor:
    from langchain_core.messages import AIMessage, ToolMessage

    executor = AgentExecutor(settings=SimpleNamespace(agent_max_steps=3))
    executor._graph = _FakeGraph([
        {"agent": {"messages": [AIMessage(
            content="先算一下",
            tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "c1"}],
        )]}},
        {"tools": {"messages": [ToolMessage(content="2", tool_call_id="c1", name="calculator")]}},
        {"agent": {"messages": [AIMessage(content="答案是 2")]}},
    ])
    return executor


def test_terminal_answer_is_not_duplicated_as_thought_step():
    events = list(_executor(None).run_stream("1+1"))

    answers = [e for e in events if e["type"] == "answer"]
    thoughts = [e for e in events if e["type"] == "step" and e["step"]["type"] == "thought"]

    assert [a["content"] for a in answers] == ["答案是 2"]
    assert thoughts == [], "终止答案不应再作为 thought 步骤推送"


def test_run_wrapper_returns_answer_and_steps():
    result = _executor(None).run("1+1")

    assert result["answer"] == "答案是 2"
    assert [s["type"] for s in result["steps"]] == ["tool_call", "observation"]


# ============================================================
# 凭据脱敏
# ============================================================
class TestMaskSecrets:
    def test_openai_style_key_is_masked(self):
        masked = mask_secrets("Error code: 401 - Incorrect API key provided: sk-abcdef123456")
        assert "sk-abcdef123456" not in masked
        assert "401" in masked  # 有价值的信息保留

    def test_bearer_token_is_masked(self):
        assert mask_secrets("Bearer abcdefghijklmn") == "***"

    def test_plain_text_untouched(self):
        assert mask_secrets("模型 gpt-4o-mini 调用失败：连接超时") == "模型 gpt-4o-mini 调用失败：连接超时"


# ============================================================
# 评测指标
# ============================================================
def test_hit_rate_denominator_only_counts_evaluable_cases(monkeypatch):
    import eval.eval_rag as ev
    from rag.vector_store import ScoredRecord

    class _Svc:
        def retrieve(self, kb_id, question, top_k=4):
            return [ScoredRecord(id="1", text="t", score=0.9,
                                 metadata={"doc_id": "doc1", "filename": "a.txt"})]

    monkeypatch.setattr(ev, "get_rag_service", lambda: _Svc())
    cases = [
        {"question": "q1", "expected_doc_id": "doc1"},
        {"question": "q2", "keywords": ["x"]},   # 没有 expected_doc_id，不应进入命中率分母
    ]

    result = ev.evaluate("kb", cases, top_k=4)

    assert result["hit_rate"] == 1.0


# ============================================================
# 工作流节点语义
# ============================================================
class TestEmptyFieldsSemantics:
    def test_empty_input_fields_exposes_nothing(self):
        output, _ = _input_node({"fields": []}, {"__inputs__": {"a": 1}}, SimpleNamespace())
        assert output == {}

    def test_absent_input_fields_exposes_everything(self):
        output, _ = _input_node({}, {"__inputs__": {"a": 1}}, SimpleNamespace())
        assert output == {"a": 1}

    def test_empty_output_fields_yields_empty_result(self):
        output, _ = _output_node({"fields": {}}, {}, SimpleNamespace())
        assert output == {}


def test_node_config_error_is_validation_error(monkeypatch):
    """节点配置错误（缺 prompt）是请求问题，应为 400 而不是 500。"""
    from workflow.executor import WorkflowExecutor
    from workflow.nodes import NodeServices

    services = NodeServices(
        settings=SimpleNamespace(llm_model="m", workflow_timeout_seconds=10),
        rag_service=None,
        agent_executor=None,
    )
    executor = WorkflowExecutor(services)
    definition = {
        "nodes": [
            {"id": "in", "type": "input"},
            {"id": "bad", "type": "llm", "config": {}},
            {"id": "out", "type": "output", "config": {"fields": {"x": "bad"}}},
        ],
        "edges": [
            {"source": "in", "target": "bad"},
            {"source": "bad", "target": "out"},
        ],
    }

    with pytest.raises(WorkflowValidationError):
        executor.run(definition, {})


# ============================================================
# Embedding 降级
# ============================================================
class TestEmbedderFallback:
    def test_self_hosted_endpoint_uses_remote_without_key(self):
        from rag.embeddings import _use_remote_endpoint

        settings = SimpleNamespace(
            openai_api_key="",
            openai_base_url="",
            embedding_base_url="http://127.0.0.1:1234/v1",
        )
        assert _use_remote_endpoint(settings) is True

    def test_public_endpoint_without_key_falls_back_to_hash(self):
        from rag.embeddings import _use_remote_endpoint

        # 旧的 _PUBLIC_HOSTS 只列了 api.openai.com，其他公开提供商会被误判为本地
        for base in ("https://api.deepseek.com/v1",
                     "https://open.bigmodel.cn/api/paas/v4",
                     "https://dashscope.aliyuncs.com/compatible-mode/v1"):
            settings = SimpleNamespace(
                openai_api_key="", openai_base_url=base, embedding_base_url="",
            )
            assert _use_remote_endpoint(settings) is False, base

    def test_empty_api_key_does_not_crash_client_construction(self):
        from rag.embeddings import OpenAIEmbedder

        settings = SimpleNamespace(
            openai_api_key="",
            openai_base_url="http://127.0.0.1:1234/v1",
            embedding_base_url="",
            embedding_model="bge-m3",
        )
        # openai SDK 对空字符串会直接抛 Missing credentials，本地端点必须可构造
        OpenAIEmbedder(settings)


# ============================================================
# 向量维度
# ============================================================
def test_dimension_mismatch_is_business_error_not_500(tmp_path):
    from rag.vector_store import LocalVectorStore, VectorRecord

    store = LocalVectorStore(tmp_path / "vdb")
    store.create_collection("kb")
    store.add("kb", [VectorRecord(id="1", text="a", embedding=[1.0, 0.0], metadata={})])

    with pytest.raises(DimensionMismatchError):
        store.search("kb", [1.0, 0.0, 0.0], top_k=1)
