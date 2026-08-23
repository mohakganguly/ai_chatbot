"""
Schemas used by the chatbot guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GuardrailResult:
    """
    Result returned by a guardrail check.
    """

    # Whether the request is allowed to continue
    allowed: bool

    # Human-readable reason for the decision
    reason: str | None = None

    # Category associated with the decision
    category: str | None = None