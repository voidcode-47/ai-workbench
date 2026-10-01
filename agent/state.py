"""Agent 的状态图状态定义。"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """messages 使用 add_messages reducer，自动按消息 id 追加/覆盖。"""

    messages: Annotated[list, add_messages]
