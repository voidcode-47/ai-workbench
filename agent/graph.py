"""基于 LangGraph 的 ReAct 状态图：agent（推理） <-> tools（行动/观察）。"""
from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.state import AgentState
from agent.tools import get_tools
from app.config import Settings

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """你是一个工具增强型 AI 助手，通过"推理 -> 行动 -> 观察"的循环完成任务。

规则：
1. 需要实时信息、计算、数据库查询、文件内容或运行代码时，必须调用对应工具，不要凭空编造；
2. 每次只选择最合适的一个工具，参数必须符合工具 schema；
3. 工具结果不含所需信息时，直接基于已有信息回答，或建议用户改用更合适的渠道（如气象网站、地图等），严禁用相同或高度相似的词反复搜索；
4. 拿到足够信息后用简体中文给出最终回答；回答必须简洁直接，只给结论和关键信息，不要复述思考过程、不要复述搜索过程、不要道歉、不要客套；
5. 同一工具连续失败 2 次应换方案或向用户说明原因，不要死循环。"""


def _make_llm(settings: Settings, *, streaming: bool):
    from langchain_openai import ChatOpenAI

    tools = get_tools(settings)
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key or "sk-no-key-required",
        base_url=settings.openai_base_url,
        temperature=0.2,
        timeout=60,
        streaming=streaming,
    )
    llm_with_tools = llm.bind_tools(tools)
    return tools, llm_with_tools


def build_agent_graph(settings: Settings):
    """编译并返回 LangGraph 可执行图。

    - 非流式路径（fallback）：agent 节点一次 invoke 出完整 AIMessage，
      用于网关不支持 SSE 时的降级路径，也用于旧的 /chat 接口。
    - 逐 token 流式路径（streaming）：agent 节点用 astream 分块产出，
      供 /chat/stream 打字机效果使用。
    """
    from langchain_core.messages import SystemMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    graph = _build_graph(settings, streaming=False)
    logger.info("Agent 非流式状态图构建完成")
    return graph


def build_agent_graph_streaming(settings: Settings):
    """编译逐 token 流式的 Agent 图。

    agent 节点通过 `astream` 逐 AIMessageChunk 产出，
    LangGraph `stream_mode="messages"` 会把每个 chunk 推给调用者。
    若该网关/模型不支持 SSE，首次调用会抛错，由 executor 捕获后
    降级到 build_agent_graph 的 fallback 路径。
    """
    graph = _build_graph(settings, streaming=True)
    logger.info("Agent 流式状态图构建完成")
    return graph


def _build_graph(settings: Settings, *, streaming: bool):
    from langchain_core.messages import SystemMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    tools, llm_with_tools = _make_llm(settings, streaming=streaming)
    system_message = SystemMessage(content=AGENT_SYSTEM_PROMPT)

    if streaming:

        async def agent_node(state: AgentState):
            """逐 token 流式推理节点。

            `astream` 产出的是 AIMessageChunk（增量）。若网关不支持
            streaming 会抛 OpenAIError/LLMError，由 executor 层统一降级。
            """
            chunks = []
            async for chunk in llm_with_tools.astream([system_message, *state["messages"]]):
                if chunk.content:
                    chunks.append(chunk)
            # 把增量 chunk 合并成单个 AIMessage，保留 tool_calls
            if not chunks:
                from langchain_core.messages import AIMessage
                return {"messages": [AIMessage(content="")]}
            merged = chunks[0]
            for extra in chunks[1:]:
                merged = merged + extra
            return {"messages": [merged]}

    else:

        def agent_node(state: AgentState) -> dict:
            response = llm_with_tools.invoke([system_message, *state["messages"]])
            return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    # tools_condition：最后一条消息含 tool_calls 则去 tools，否则结束
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    logger.info("Agent 状态图构建完成，streaming=%s，工具数=%d", streaming, len(tools))
    return graph.compile()
