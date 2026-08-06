"""
registry.py

Central registry for all tools available to the agent.
"""

from __future__ import annotations

from typing import Dict

from tools.base import BaseTool
from tools.retriever_tool import RetrieverTool
from tools.calculator_tool import CalculatorTool
from tools.web_search_tool import WebSearchTool
from tools.python_tool import PythonTool


class ToolRegistry:
    """
    Stores and provides access to all available tools.

    The planner never creates tool instances directly.
    It only requests them from this registry.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(
        self,
        tool: BaseTool,
    ) -> None:
        """
        Register a tool.
        """

        self._tools[tool.name] = tool

    def get(
        self,
        name: str,
    ) -> BaseTool:
        """
        Retrieve a tool by name.

        Raises
        ------
        KeyError
            If the tool is not registered.
        """

        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' is not registered."
            )

        return self._tools[name]

    def exists(
        self,
        name: str,
    ) -> bool:
        """
        Check whether a tool exists.
        """

        return name in self._tools

    def list_tools(
        self,
    ) -> list[BaseTool]:
        """
        Return all registered tool instances.
        """

        return list(self._tools.values())

    def list_tool_metadata(
        self,
    ) -> list[dict]:
        """
        Return planner-friendly metadata.
        """

        return [
            {
                "name": tool.name,
                "description": tool.description,
            }
            for tool in self._tools.values()
        ]


# ==========================================================
# Global Registry
# ==========================================================

registry = ToolRegistry()

registry.register(
    RetrieverTool()
)

registry.register(
    CalculatorTool()
)

registry.register(
    WebSearchTool()
)

registry.register(
    PythonTool()
)