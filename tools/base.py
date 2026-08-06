"""
base.py

Base interfaces shared by every tool.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


# ==========================================================
# Tool Context
# ==========================================================

class ToolContext(BaseModel):
    """
    Runtime context supplied to every tool.

    The executor constructs this object before
    invoking a tool. It contains everything a
    tool may need during execution.
    """

    # Original user query
    query: str

    # Conversation history
    messages: list[BaseMessage] = Field(
        default_factory=list,
    )

    # Tool-specific arguments produced by the planner
    tool_input: dict[str, Any] = Field(
        default_factory=dict,
    )

    # Additional runtime metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ==========================================================
# Tool Result
# ==========================================================

class ToolResult(BaseModel):
    """
    Standard output returned by every tool.
    """

    # Tool that generated this result
    tool_name: str

    # Whether execution succeeded
    success: bool

    # Machine-readable output
    output: Any | None = None

    # Human-readable summary for the Answer Node
    summary: str

    # Error message if execution failed
    error: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ==========================================================
# Base Tool
# ==========================================================

class BaseTool(ABC):
    """
    Abstract base class for every tool.

    Every tool must expose:
    - name
    - description
    - invoke()
    """

    name: str

    description: str

    @abstractmethod
    def invoke(
        self,
        context: ToolContext,
    ) -> ToolResult:
        """
        Execute the tool.

        Parameters
        ----------
        context
            Runtime information supplied by
            the executor.

        Returns
        -------
        ToolResult
            Standardized tool response.
        """
        raise NotImplementedError