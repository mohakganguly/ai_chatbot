"""
executor.py

Executes tool calls produced by the planner.
"""

from __future__ import annotations

from graph.state import AgentState

from agents.schemas import ToolCall
from agents.registry import registry

from tools.base import (
    ToolContext,
    ToolResult,
)

from utils.logger import get_logger
from observability.tracing import trace_node

logger = get_logger(__name__)


class Executor:
    """
    Executes ToolCall objects.

    The executor is responsible for
    providing runtime context to tools.
    """

    def execute(
        self,
        tool_call: ToolCall,
        state: AgentState,
    ) -> ToolResult:

        with trace_node(
            "executor",
            tool=tool_call.tool,
            tool_input=tool_call.tool_input,
            iteration=state["iteration"],
        ):

            # --------------------------------------------------
            # Tool Lookup
            # --------------------------------------------------

            with trace_node("executor.tool_lookup"):

                try:

                    tool = registry.get(
                        tool_call.tool
                    )

                except KeyError as exc:

                    logger.exception(
                        "Tool '%s' not found.",
                        tool_call.tool,
                    )

                    return ToolResult(

                        tool_name=tool_call.tool,

                        success=False,

                        output=None,

                        summary="Requested tool was not found.",

                        error=str(exc),
                    )

            # --------------------------------------------------
            # Build Runtime Context
            # --------------------------------------------------

            with trace_node("executor.build_context"):

                context = ToolContext(

                    query=state["query"],

                    messages=state["messages"][-6:],

                    metadata=state["metadata"],

                    tool_input=tool_call.tool_input,
                )

            logger.info(
                "Executing tool: %s",
                tool_call.tool,
            )

            # --------------------------------------------------
            # Execute Tool
            # --------------------------------------------------

            with trace_node(
                "executor.execute_tool",
                tool=tool_call.tool,
            ):

                result = tool.invoke(
                    context
                )

            # --------------------------------------------------
            # Return Result
            # --------------------------------------------------

            with trace_node(
                "executor.result",
                success=result.success,
            ):

                logger.info(
                    "Tool '%s' completed. Success=%s",
                    tool_call.tool,
                    result.success,
                )

                return result