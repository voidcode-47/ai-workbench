"""WorkflowEngine 单元测试：定义校验、环检测、拓扑执行、条件分支与超时。"""
from __future__ import annotations

import time

import pytest

from app.exceptions import WorkflowExecutionError, WorkflowValidationError
from workflow.engine import WorkflowEngine


def _node(nid: str, ntype: str, config: dict | None = None) -> dict:
    node = {"id": nid, "type": ntype}
    if config is not None:
        node["config"] = config
    return node


def _edge(source: str, target: str, label: str | None = None) -> dict:
    edge = {"source": source, "target": target}
    if label is not None:
        edge["label"] = label
    return edge


def _runner(outputs: dict):
    """假 runner：按节点 id 返回预设输出。"""

    def runner(node: dict, context: dict):
        return outputs[node["id"]], {"scheduled": True}

    return runner


@pytest.fixture
def engine() -> WorkflowEngine:
    return WorkflowEngine()


class TestValidation:
    def test_rejects_empty_nodes(self, engine):
        with pytest.raises(WorkflowValidationError, match="nodes"):
            engine.validate({"nodes": [], "edges": []})

    def test_rejects_missing_node_id(self, engine):
        with pytest.raises(WorkflowValidationError, match="缺少 id"):
            engine.validate({"nodes": [{"type": "input"}]})

    def test_rejects_duplicate_node_id(self, engine):
        with pytest.raises(WorkflowValidationError, match="重复"):
            engine.validate({"nodes": [_node("a", "input"), _node("a", "output")]})

    def test_rejects_unknown_node_type(self, engine):
        with pytest.raises(WorkflowValidationError, match="类型非法"):
            engine.validate({"nodes": [_node("a", "transformer")]})

    def test_rejects_edge_referencing_missing_node(self, engine):
        with pytest.raises(WorkflowValidationError, match="不存在的节点"):
            engine.validate({
                "nodes": [_node("a", "input"), _node("b", "output")],
                "edges": [_edge("a", "x")],
            })

    def test_rejects_self_loop(self, engine):
        with pytest.raises(WorkflowValidationError, match="自环"):
            engine.validate({
                "nodes": [_node("a", "input"), _node("b", "output")],
                "edges": [_edge("a", "a"), _edge("a", "b")],
            })

    def test_rejects_cycle(self, engine):
        with pytest.raises(WorkflowValidationError, match="环"):
            engine.validate({
                "nodes": [_node("a", "input"), _node("b", "llm"), _node("c", "llm")],
                "edges": [_edge("a", "b"), _edge("b", "c"), _edge("c", "b")],
            })

    def test_requires_output_node(self, engine):
        with pytest.raises(WorkflowValidationError, match="output"):
            engine.validate({
                "nodes": [_node("a", "input"), _node("b", "llm")],
                "edges": [_edge("a", "b")],
            })

    def test_valid_linear_definition_returns_topo_order(self, engine):
        parsed = engine.validate({
            "nodes": [
                _node("in", "input"),
                _node("mid", "llm"),
                _node("out", "output"),
            ],
            "edges": [_edge("in", "mid"), _edge("mid", "out")],
        })
        assert parsed["topo_order"] == ["in", "mid", "out"]
        assert parsed["entry_nodes"] == ["in"]
        assert parsed["output_nodes"] == ["out"]


class TestExecution:
    def test_linear_flow_collects_output_node_result(self, engine):
        definition = {
            "nodes": [
                _node("in", "input"),
                _node("llm", "llm"),
                _node("out", "output", {"fields": {"answer": "llm"}}),
            ],
            "edges": [_edge("in", "llm"), _edge("llm", "out")],
        }
        parsed = engine.validate(definition)
        outputs = {"in": {"q": "你好"}, "llm": "答案", "out": {"answer": "答案"}}
        result = engine.execute(parsed, {"q": "你好"}, _runner(outputs))
        assert result["outputs"] == {"answer": "答案"}
        statuses = {log["node"]: log["status"] for log in result["logs"]}
        assert statuses == {"in": "success", "llm": "success", "out": "success"}

    def test_condition_activates_only_matched_branch(self, engine):
        definition = {
            "nodes": [
                _node("in", "input"),
                _node("cond", "condition"),
                _node("yes", "llm"),
                _node("no", "llm"),
                _node("out", "output", {"fields": {"picked": "yes", "skipped": "no"}}),
            ],
            "edges": [
                _edge("in", "cond"),
                _edge("cond", "yes", "yes"),
                _edge("cond", "no", "no"),
                _edge("yes", "out"),
                _edge("no", "out"),
            ],
        }
        parsed = engine.validate(definition)
        outputs = {
            "in": {},
            "cond": "yes",
            "yes": "走了 yes",
            "no": "走了 no",
            "out": {"picked": "走了 yes", "skipped": ""},
        }
        result = engine.execute(parsed, {}, _runner(outputs))

        statuses = {log["node"]: log["status"] for log in result["logs"]}
        assert statuses["yes"] == "success"
        assert statuses["no"] == "skipped"
        # 未执行的分支不产生输出，output 节点解析不到时为空串
        assert result["outputs"] == {"picked": "走了 yes", "skipped": ""}

    def test_unmatched_condition_label_falls_back_to_default_edge(self, engine):
        definition = {
            "nodes": [
                _node("in", "input"),
                _node("cond", "condition"),
                _node("other", "llm"),
                _node("out", "output"),
            ],
            "edges": [
                _edge("in", "cond"),
                _edge("cond", "other", "unexpected-label"),
                _edge("cond", "out", "default"),
            ],
        }
        parsed = engine.validate(definition)
        outputs = {"in": {}, "cond": "yes", "other": "x", "out": {"done": True}}
        result = engine.execute(parsed, {}, _runner(outputs))
        statuses = {log["node"]: log["status"] for log in result["logs"]}
        assert statuses["other"] == "skipped"
        assert statuses["out"] == "success"

    def test_labeled_edges_from_normal_node_do_not_activate(self, engine):
        definition = {
            "nodes": [
                _node("in", "input"),
                _node("out", "output"),
            ],
            "edges": [_edge("in", "out", "conditional-label")],
        }
        parsed = engine.validate(definition)
        result = engine.execute(parsed, {}, _runner({"in": {}}))
        # 普通节点不会激活带 label 的边，output 被跳过，输出为空
        assert result["outputs"] == {}
        assert result["logs"][1]["status"] == "skipped"

    def test_runner_failure_is_logged_and_reraised(self, engine):
        definition = {
            "nodes": [_node("in", "input"), _node("out", "output")],
            "edges": [_edge("in", "out")],
        }
        parsed = engine.validate(definition)

        def failing_runner(node, context):
            if node["id"] == "in":
                raise RuntimeError("boom")
            return "x", {}

        with pytest.raises(RuntimeError, match="boom"):
            engine.execute(parsed, {}, failing_runner)

    def test_timeout_between_nodes_aborts_execution(self, engine):
        definition = {
            "nodes": [
                _node("slow", "input"),
                _node("out", "output"),
            ],
            "edges": [_edge("slow", "out")],
        }
        parsed = engine.validate(definition)

        def slow_runner(node, context):
            if node["id"] == "slow":
                time.sleep(0.05)
            return "x", {}

        with pytest.raises(WorkflowExecutionError, match="超时"):
            engine.execute(parsed, {}, slow_runner, timeout=0.01)
