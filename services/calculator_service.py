"""
calculator_service.py

Simple calculator service.
"""

from __future__ import annotations

import math


class CalculatorService:
    """
    Performs mathematical calculations.
    """

    SAFE_GLOBALS = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sqrt": math.sqrt,
        "ceil": math.ceil,
        "floor": math.floor,
    }

    def calculate(
        self,
        expression: str,
    ) -> float:

        return eval(
            expression,
            self.SAFE_GLOBALS,
            {},
        )