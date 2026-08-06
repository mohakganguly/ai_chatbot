"""
router.py

Routing helpers for LangGraph.
"""

from __future__ import annotations

from graph.state import AgentState

MAX_ITERATIONS = 3

def should_continue(
    state: AgentState,
) -> str:
    """
    Decide whether the agent should
    continue executing tools or
    generate the final answer.
    """
    if state["iteration"] >= MAX_ITERATIONS:
        return "finish"
    observation = state["observation"]

    if observation is None:
        return "finish"

    if observation.continue_execution:
        return "continue"

    return "finish"