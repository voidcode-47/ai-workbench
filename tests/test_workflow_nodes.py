"""workflow.nodes 单元测试：模板渲染与各类型节点处理函数（不调用真实 LLM）。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from workflow.nodes import (
    _condition_node,
    _input_node,
    _llm_node,
    _output_node,
    _retrieval_node,
    _tool_node,
    render,
)


@pytest.fixture
def context() -> dict:
    return {
        "__inputs__": {"topic": "RAG", "count": 3},
        "n1": {"output": "上游输出"},
        "n2": {"output": {"title": "标题", "nested": {"deep": "深层"}}},
        "missing": {},
    }


class TestRender:
    def test_renders_inputs_variable(self, context):
        assert render("主题：{{inputs.topic}}", context) == "主题：RAG"

    def test_renders_node_output(self, context):
        assert render("结果={{n1}}", context) == "结果=上游输出"

    def test_renders_dict_attribute_path(self, context):
        assert render("{{n2.title}}", context) == "标题"

    def test_missing_variable_is_replaced_with_empty_string(self, context):
        assert render("[{{inputs.absent}}]", context) == "[]"
        assert render("[{{ghost}}]", context) == "[]"

    def test_recurses_into_dict_and_list(self, context):
        value = {"prompt": ["{{inputs.topic}}", 1, {"k": "{{n2.title}}"}]}
        assert render(value, context) == {"prompt": ["RAG", 1, {"k": "标题"}]}

    def test_non_string_values_pass_through(self, context):
        assert render(42, context) == 42
        assert render(None, context) is None


class TestNodeHandlers:
    def test_input_node_filters_to_provided_fields(self):
        output, meta = _input_node(
            {"fields": ["topic", "absent"]}, {"__inputs__": {"topic": "RAG"}}, None
        )
        assert output == {"topic": "RAG"}
        assert meta == {"fields": ["topic"]}

    def test_input_node_without_fields_returns_all_inputs(self):
        output, _ = _input_node({}, {"__inputs__": {"a": 1}}, None)
        assert output == {"a": 1}

    def test_condition_node_matches_configured_route(self):
        config = {"field": "inputs.mode", "routes": {"branch_a": "x", "branch_b": "y"}}
        label, meta = _condition_node(config, {"__inputs__": {"mode": "y"}}, None)
        assert label == "branch_b"
        assert meta["branch"] == "branch_b"

    def test_condition_node_falls_back_to_default(self):
        config = {
            "field": "inputs.mode",
            "routes": {"branch_a": "x"},
            "default": "fallback",
        }
        label, _ = _condition_node(config, {"__inputs__": {"mode": "z"}}, None)
        assert label == "fallback"

    def test_condition_node_requires_field_and_routes(self):
        with pytest.raises(ValueError, match="field 或 config.routes"):
            _condition_node({"routes": {"a": "b"}}, {}, None)

    def test_output_node_resolves_field_references(self):
        context = {"llm": {"output": "最终答案"}}
        output, meta = _output_node({"fields": {"answer": "llm"}}, context, None)
        assert output == {"answer": "最终答案"}
        assert meta == {"keys": ["answer"]}

    def test_llm_node_requires_prompt(self):
        with pytest.raises(ValueError, match="prompt"):
            _llm_node({}, {}, None)

    def test_retrieval_node_requires_kb_id_and_question(self):
        with pytest.raises(ValueError, match="kb_id"):
            _retrieval_node({"question": "q"}, {}, None)

    def test_tool_node_rejects_non_whitelisted_tools(self):
        # run_python 是任意代码执行工具，永远不允许通过工作流调用
        with pytest.raises(ValueError, match="安全工具"):
            _tool_node({"name": "run_python"}, {}, None)
        with pytest.raises(ValueError, match="安全工具"):
            _tool_node({"name": "read_file"}, {}, None)

    def test_tool_node_invokes_safe_argumentless_tool(self):
        services = SimpleNamespace(settings=SimpleNamespace(enable_python_tool=False))
        output, meta = _tool_node({"name": "current_date"}, {}, services)
        assert "星期" in output
        assert meta == {"tool": "current_date"}
