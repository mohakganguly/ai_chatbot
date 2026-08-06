"""
router_schema.py

Schema for the router output.
"""

from typing import Literal

from pydantic import BaseModel


class RouterOutput(BaseModel):

    route: Literal[
        "general",
        "rag",
    ]