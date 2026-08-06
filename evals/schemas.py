"""
Shared evaluation schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evals.metrics import EvaluationMetrics


@dataclass
class EvaluationSample:
    """
    Single benchmark sample.
    """

    question: str

    answer: str

    contexts: list[str]

    citations: list[Any]

    latency: float

    ground_truth: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """
    Output of an evaluation run.
    """

    sample: EvaluationSample

    metrics: EvaluationMetrics