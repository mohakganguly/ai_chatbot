"""
metrics.py

Enterprise evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """
    Metrics collected during an evaluation run.
    """

    # ---------------------------
    # RAGAS Metrics
    # ---------------------------

    faithfulness: float = 0.0

    answer_relevancy: float = 0.0

    context_precision: float = 0.0

    context_recall: float = 0.0

    # ---------------------------
    # Enterprise Metrics
    # ---------------------------

    latency: float = 0.0

    planner_iterations: int = 0

    tool_calls: int = 0

    citation_count: int = 0