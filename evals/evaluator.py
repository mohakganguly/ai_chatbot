"""
evaluator.py

Enterprise evaluation orchestrator.

Coordinates the complete evaluation workflow.

Pipeline

Question(s)
      ↓
Collector
      ↓
EvaluationSample(s)
      ↓
RAGAS Runner
      ↓
EvaluationResult(s)
"""

from __future__ import annotations

from evals.collector import Collector
from evals.ragas_runner import RagasRunner

from evals.schemas import (
    EvaluationResult,
    EvaluationSample,
)


class Evaluator:
    """
    Coordinates the complete evaluation pipeline.
    """

    def __init__(self):

        self.collector = Collector()

        self.ragas = RagasRunner()

    # ---------------------------------------------------------
    # Evaluate a single question
    # ---------------------------------------------------------

    def evaluate(
            self,
            question: str,
            ground_truth: str | None = None,
    ) -> EvaluationResult:

        sample = self.collector.collect(
            question
        )
        sample.ground_truth = ground_truth

        results = self.ragas.evaluate(
            [sample]
        )

        return results[0]

    # ---------------------------------------------------------
    # Evaluate multiple questions
    # ---------------------------------------------------------

    def evaluate_batch(
        self,
        questions: list[str],
    ) -> list[EvaluationResult]:

        samples: list[EvaluationSample] = [

            self.collector.collect(question)

            for question in questions

        ]

        return self.ragas.evaluate(
            samples
        )