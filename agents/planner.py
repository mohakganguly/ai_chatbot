"""
Planning agent responsible for selecting
the next tool.
"""

from __future__ import annotations

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from graph.state import AgentState
from agents.schemas import ToolCall

from agents.prompts import (
    PLANNER_SYSTEM_PROMPT,
    build_tool_descriptions,
    build_planner_context,
)

from agents.registry import registry
from core.llm import planner_llm

from utils.logger import get_logger
from observability.tracing import trace_node


logger = get_logger(__name__)


class Planner:
    """
    Determines the next tool to execute.
    """

    def __init__(self):

        self.llm = planner_llm.with_structured_output(
            ToolCall,
            method="json_schema",
        )

    def plan(
        self,
        state: AgentState,
    ) -> ToolCall|None:

        with trace_node(
            "Planner",
            iteration=state["iteration"],
            previous_tools=len(state["tool_history"]),
        ):

            # --------------------------------------------------
            # Build Prompt
            # --------------------------------------------------

            with trace_node("Planner - Build Prompt"):

                tool_descriptions = build_tool_descriptions(
                    registry.list_tool_metadata()
                )

                system_prompt = PLANNER_SYSTEM_PROMPT.format(
                    tool_descriptions=tool_descriptions,
                )

                planner_context = build_planner_context(
                    state
                )

                messages = [

                    SystemMessage(
                        content=system_prompt,
                    ),

                    HumanMessage(
                        content=planner_context,
                    ),

                ]

            logger.info(
                "Running Planner"
            )

            try:

                # --------------------------------------------------
                # Planner LLM
                # --------------------------------------------------

                with trace_node(
                    "Planner - LLM"
                ):

                    tool_call = self.llm.invoke(
                        messages
                    )

                # --------------------------------------------------
                # Planner Validation
                # --------------------------------------------------

                with trace_node(
                    "Planner - Validation"
                ):

                    if state["tool_history"]:

                        last_tool = state["tool_history"][-1]

                        if (
                            last_tool.tool
                            == tool_call.tool
                        ):

                            if state["tool_result"]:

                                if state["observation"]:

                                    state[
                                        "observation"
                                    ].continue_execution = False

                logger.info(
                    "Planner selected tool: %s",
                    tool_call.tool,
                )

                return tool_call

            except Exception as e:

                logger.exception(
                    "Planner failed to generate structured output: %s",
                    str(e),
                )

                return None