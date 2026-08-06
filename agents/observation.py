"""
observation.py

Observation agent.
"""

from __future__ import annotations

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from graph.state import AgentState

from agents.prompts import (
    OBSERVATION_SYSTEM_PROMPT,
)

from agents.schemas import (
    ObservationDecision,
)

from core.llm import observation_llm

from observability.tracing import trace_node

from utils.logger import get_logger

logger = get_logger(__name__)


class Observation:
    """
    Determines whether more tools
    are required.
    """

    def __init__(self):

        self.llm = observation_llm.with_structured_output(
            ObservationDecision
        )

    def observe(
        self,
        state: AgentState,
    ) -> ObservationDecision:

        with trace_node(
            "observation.decision",
            iteration=state["iteration"],
        ):

            # -----------------------------------------
            # Build Prompt
            # -----------------------------------------

            with trace_node("observation.build_prompt"):

                tool_result = ""

                if state["tool_result"]:

                    tool_result = str(
                        state["tool_result"].output
                    )

                messages = [

                    SystemMessage(
                        content=OBSERVATION_SYSTEM_PROMPT
                    ),

                    HumanMessage(
                        content=f"""
Original User Request

{state["query"]}

----------------------------------

Latest Tool Result

{tool_result}
"""
                    ),
                ]

            logger.info("Running Observation Agent")

            # -----------------------------------------
            # LLM
            # -----------------------------------------

            with trace_node("observation.llm"):

                decision = self.llm.invoke(
                    messages
                )

            # -----------------------------------------
            # Decision
            # -----------------------------------------

            with trace_node(
                "observation.decision",
                continue_execution=decision.continue_execution,
            ):

                logger.info(
                    "Observation decision: continue_execution=%s",
                    decision.continue_execution,
                )

                return decision