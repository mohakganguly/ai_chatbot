"""
python_service.py

Executes Python code in a controlled environment.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd


class PythonService:
    """
    Execute Python snippets.
    """

    def execute(
        self,
        code: str,
    ) -> str:

        output = io.StringIO()

        globals_dict = {

            "__builtins__": {},

            "pd": pd,

            "np": np,

            "len": len,

            "sum": sum,

            "min": min,

            "max": max,

            "abs": abs,

            "round": round,
        }

        with contextlib.redirect_stdout(
            output
        ):

            exec(
                code,
                globals_dict,
                {},
            )

        return output.getvalue()