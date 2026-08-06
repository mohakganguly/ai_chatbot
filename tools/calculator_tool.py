"""
calculator_tool.py
"""

from __future__ import annotations

from tools.base import (
    BaseTool,
    ToolContext,
    ToolResult,
)

from services.calculator_service import (
    CalculatorService,
)


class CalculatorTool(BaseTool):

    name = "calculator"

    description = (
        "Perform mathematical calculations including "
        "arithmetic, percentages, averages, ratios, "
        "and formulas."
    )

    def __init__(self):

        self.service = CalculatorService()

    def invoke(
        self,
        context: ToolContext,
    ) -> ToolResult:

        expression = context.tool_input.get(
            "expression"
        )

        if not expression:

            return ToolResult(

                tool_name=self.name,

                success=False,

                summary="No expression provided.",

                error="Missing expression.",
            )

        try:

            result = self.service.calculate(
                expression
            )

            return ToolResult(

                tool_name=self.name,

                success=True,

                output=result,

                summary=f"Calculated result: {result}",
            )

        except Exception as exc:

            return ToolResult(

                tool_name=self.name,

                success=False,

                summary="Calculation failed.",

                error=str(exc),
            )