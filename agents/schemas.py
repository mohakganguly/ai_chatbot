"""
schemas.py

Shared schemas used by the planning and observation agents.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ==========================================================
# Planner
# ==========================================================

class ToolCall(BaseModel):
    """
    Represents a tool selected by the planner.
    """

    tool: str = Field(
        description="Name of the selected tool."
    )

    reason: str = Field(
        description="Reason for selecting this tool."
    )

    tool_input: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool."
    )

    confidence: float | None = Field(
        default=None,
        description="Planner confidence."
    )


# ==========================================================
# Observation
# ==========================================================

class ObservationDecision(BaseModel):
    """
    Decision made after observing
    the latest tool execution.
    """

    continue_execution: bool = Field(
        description="Whether another planning step is required."
    )

    reason: str = Field(
        description="Reason for the decision."
    )

    next_query: str | None = Field(
        default=None,
        description="Query for the next planning step."
    )

    suggested_tool: str | None = Field(
        default=None,
        description="Suggested next tool if applicable."
    )