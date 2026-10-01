"""calculator 安全计算器工具的单元测试。

契约：工具无论输入是否合法都必须返回字符串，绝不向外抛异常
（返回值会直接进入 Agent 的工具观察结果）。
"""
from __future__ import annotations

import time

from agent.tools import calculator


def test_basic_arithmetic():
    assert calculator.invoke({"expression": "(3 + 4) * 2 / 5"}) == "2.8"


def test_floor_div_modulo_and_normal_power():
    assert calculator.invoke({"expression": "7 // 2"}) == "3"
    assert calculator.invoke({"expression": "7 % 3"}) == "1"
    assert calculator.invoke({"expression": "2 ** 10"}) == "1024"


def test_unary_minus_and_float():
    assert calculator.invoke({"expression": "-2 + 0.5"}) == "-1.5"


def test_rejects_non_numeric_identifiers():
    assert calculator.invoke({"expression": "abc"}).startswith("计算失败")


def test_rejects_function_calls():
    assert calculator.invoke({"expression": "__import__('os')"}).startswith("计算失败")


def test_division_by_zero_is_reported_not_raised():
    assert calculator.invoke({"expression": "1 / 0"}).startswith("计算失败")


def test_large_exponent_is_rejected_quickly():
    # 无指数上限时：大整数 str() 在 Python 3.11 会触发 4300 位限制异常，
    # 且计算本身可能耗尽 CPU/内存。工具必须快速、安全地返回失败字符串。
    started = time.monotonic()
    result = calculator.invoke({"expression": "2 ** 200000"})
    elapsed = time.monotonic() - started
    assert result.startswith("计算失败")
    assert elapsed < 1.0


def test_exponent_at_limit_is_allowed():
    assert calculator.invoke({"expression": "2 ** 1000"}) == str(2 ** 1000)


def test_huge_result_string_conversion_error_is_reported_not_raised():
    # 指数在上限内但结果仍超 Python 整数转字符串位数限制时，工具同样不能抛异常
    result = calculator.invoke({"expression": "999999 ** 1000"})
    assert isinstance(result, str)
    assert result.startswith("计算失败")
