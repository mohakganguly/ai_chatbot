"""
python_tool.py
"""

from __future__ import annotations

from services.python_service import PythonService

from tools.base import (
    BaseTool,
    ToolContext,
    ToolResult,
)


class PythonTool(BaseTool):

    name = "python"

    description = (
        "Execute Python code for data analysis, "
        "statistics and dataframe operations."
    )

    def __init__(self):

        self.service = PythonService()

    def invoke(
        self,
        context: ToolContext,
    ) -> ToolResult:

        code = context.tool_input.get(
            "code"
        )

        if not code:

            return ToolResult(

                tool_name=self.name,

                success=False,

                summary="No Python code provided.",

                error="Missing code.",
            )

        try:

            result = self.service.execute(
                code
            )

            return ToolResult(

                tool_name=self.name,

                success=True,

                output=result,

                summary=result,
            )

        except Exception as exc:

            return ToolResult(

                tool_name=self.name,

                success=False,

                summary="Python execution failed.",

                error=str(exc),
            )