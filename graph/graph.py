"""
graph.py

Enterprise AI Assistant workflow.
"""

from __future__ import annotations

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from graph.state import AgentState

from graph.routes import (
    GENERAL,
    AGENT,
)

from graph.nodes import (
    input_guardrail_node,
    router_node,
    general_chat_node,
    planner_node,
    tool_guardrail_node,
    executor_node,
    observation_node,
    answer_node,
)

from graph.router import should_continue

from db.checkpoint import checkpointer

from utils.logger import get_logger

logger = get_logger(__name__)

logger.info("Building Enterprise AI Assistant Workflow...")

def route_after_planner(state: AgentState):

    tool_call = state.get("tool_call")

    if tool_call is None:
        return "answer"

    tool_name = getattr(
        tool_call,
        "tool",
        None,
    )

    if tool_name is None or tool_name.lower() == "none":
        return "answer"

    return "executor"
# ==========================================================
# Graph Builder
# ==========================================================

builder = StateGraph(AgentState)

# ==========================================================
# Nodes
# ==========================================================

builder.add_node(
    "input_guardrail",
    input_guardrail_node,
)

builder.add_node(
    "router",
    router_node,
)

builder.add_node(
    GENERAL,
    general_chat_node,
)

builder.add_node(
    "planner",
    planner_node,
)
builder.add_node(
    "tool_guardrail",
    tool_guardrail_node,
)
builder.add_node(
    "executor",
    executor_node,
)

builder.add_node(
    "observation",
    observation_node,
)

builder.add_node(
    "answer",
    answer_node,
)

# ==========================================================
# Entry
# ==========================================================

builder.add_edge(
    START,
    "input_guardrail",
)

# ==========================================================
# Input Guardrail
# ==========================================================

builder.add_conditional_edges(
    "input_guardrail",
    lambda state: state["guardrail_status"],
    {
        "SAFE": "router",
        "BLOCKED": END,
    },
)
# ==========================================================
# Router
# ==========================================================

builder.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        GENERAL: GENERAL,
        AGENT: "planner",
    },
)

builder.add_edge(
    GENERAL,
    END,
)

# ==========================================================
# Agent Loop
# ==========================================================

# builder.add_edge(
#     "planner",
#     "executor",
# )
builder.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "executor": "tool_guardrail",
        "answer": "answer",
    },
)
builder.add_conditional_edges(
    "tool_guardrail",
    lambda state: state["tool_guardrail_status"],
    {
        "SAFE": "executor",
        "BLOCKED": "observation",
    },
)
builder.add_edge(
    "executor",
    "observation",
)

builder.add_conditional_edges(
    "observation",
    should_continue,
    {
        "continue": "planner",
        "finish": "answer",
    },
)

builder.add_edge(
    "answer",
    END,
)

# ==========================================================
# Compile
# ==========================================================

chatbot = builder.compile(
    checkpointer=checkpointer,
)

logger.info(
    "Enterprise AI Assistant compiled successfully."
)