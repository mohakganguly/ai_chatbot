"""
graph.py

Builds the Enterprise AI Assistant workflow using LangGraph.
"""

from __future__ import annotations

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from graph.state import ChatState
from graph.routes import (
    GENERAL,
    RAG,
)

from graph.nodes import (
    router_node,
    general_chat_node,
    rewrite_node,
    retrieval_node,
    generation_node,
)

from db.checkpoint import checkpointer

from utils.logger import get_logger

logger = get_logger(__name__)


# ==========================================================
# Route Selector
# ==========================================================

def route_selector(
    state: ChatState,
) -> str:
    """
    Reads the route chosen by the router node.
    """

    route = state["route"]

    logger.info(
        "Routing workflow to '%s'",
        route,
    )

    return route


# ==========================================================
# Build Graph
# ==========================================================

logger.info("Building LangGraph workflow")

builder = StateGraph(ChatState)

# ----------------------------------------------------------
# Nodes
# ----------------------------------------------------------

builder.add_node(
    "router",
    router_node,
)

builder.add_node(
    GENERAL,
    general_chat_node,
)

builder.add_node(
    "rewrite",
    rewrite_node,
)

builder.add_node(
    "retrieval",
    retrieval_node,
)

builder.add_node(
    "generation",
    generation_node,
)

# ----------------------------------------------------------
# Edges
# ----------------------------------------------------------

builder.add_edge(
    START,
    "router",
)

builder.add_conditional_edges(
    "router",
    route_selector,
    {
        GENERAL: GENERAL,
        RAG: "rewrite",
    },
)

builder.add_edge(
    GENERAL,
    END,
)

builder.add_edge(
    "rewrite",
    "retrieval",
)

builder.add_edge(
    "retrieval",
    "generation",
)

builder.add_edge(
    "generation",
    END,
)

# ----------------------------------------------------------
# Compile
# ----------------------------------------------------------

rag_graph = builder.compile(
    checkpointer=checkpointer,
)

logger.info("LangGraph compiled successfully.")