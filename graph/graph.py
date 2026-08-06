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
    router_node,
    general_chat_node,
    planner_node,
    executor_node,
    observation_node,
    answer_node,
)

from graph.router import should_continue

from db.checkpoint import checkpointer

from utils.logger import get_logger

logger = get_logger(__name__)

logger.info("Building Enterprise AI Assistant Workflow...")

# ==========================================================
# Graph Builder
# ==========================================================

builder = StateGraph(AgentState)

# ==========================================================
# Nodes
# ==========================================================

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
    "router",
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

builder.add_edge(
    "planner",
    "executor",
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