"""
report.py

Enterprise evaluation report generator.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from evals.schemas import EvaluationResult

from utils.logger import get_logger


logger = get_logger(__name__)


class ReportGenerator:
    """
    Generates evaluation reports.
    """

    def __init__(self):

        self.output_dir = Path(
            "evals/reports"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ==========================================================
    # Aggregate Metrics
    # ==========================================================

    @staticmethod
    def _average(
        values: list[float],
    ) -> float:

        if not values:
            return 0.0

        return sum(values) / len(values)

    # ==========================================================
    # Summary
    # ==========================================================

    def _summary(
        self,
        results: list[EvaluationResult],
    ) -> dict:

        if not results:

            return {
                "questions": 0,
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "latency": 0.0,
                "planner_iterations": 0.0,
                "tool_calls": 0.0,
                "citation_count": 0.0,
                "retrieved_documents": 0.0,
                "retriever_usage_rate": 0.0,
            }

        metrics = [
            result.metrics
            for result in results
        ]

        # ------------------------------------------------------
        # RAGAS Metrics
        # ------------------------------------------------------

        faithfulness = self._average(
            [
                m.faithfulness
                for m in metrics
            ]
        )

        answer_relevancy = self._average(
            [
                m.answer_relevancy
                for m in metrics
            ]
        )

        context_precision = self._average(
            [
                m.context_precision
                for m in metrics
            ]
        )

        context_recall = self._average(
            [
                m.context_recall
                for m in metrics
            ]
        )

        # ------------------------------------------------------
        # Enterprise Metrics
        # ------------------------------------------------------

        latency = self._average(
            [
                m.latency
                for m in metrics
            ]
        )

        planner_iterations = self._average(
            [
                m.planner_iterations
                for m in metrics
            ]
        )

        tool_calls = self._average(
            [
                m.tool_calls
                for m in metrics
            ]
        )

        citation_count = self._average(
            [
                m.citation_count
                for m in metrics
            ]
        )

        retrieved_documents = self._average(
            [
                m.retrieved_documents
                for m in metrics
            ]
        )

        # ------------------------------------------------------
        # Retriever Usage
        # ------------------------------------------------------

        retriever_usage_rate = (
            sum(
                1
                for m in metrics
                if m.retriever_used
            )
            / len(metrics)
        )

        return {

            "questions": len(results),

            # RAGAS
            "faithfulness": faithfulness,

            "answer_relevancy": answer_relevancy,

            "context_precision": context_precision,

            "context_recall": context_recall,

            # Enterprise
            "latency": latency,

            "planner_iterations": (
                planner_iterations
            ),

            "tool_calls": tool_calls,

            "citation_count": citation_count,

            "retrieved_documents": (
                retrieved_documents
            ),

            "retriever_usage_rate": (
                retriever_usage_rate
            ),
        }

    # ==========================================================
    # Console Report
    # ==========================================================

    def print_report(
        self,
        results: list[EvaluationResult],
    ):

        summary = self._summary(
            results
        )

        print()

        print("=" * 60)

        print(
            "Enterprise AI Evaluation Report"
        )

        print("=" * 60)

        print()

        print(
            f"Questions             : "
            f"{summary['questions']}"
        )

        print(
            f"Faithfulness          : "
            f"{summary['faithfulness']:.3f}"
        )

        print(
            f"Answer Relevancy      : "
            f"{summary['answer_relevancy']:.3f}"
        )

        print(
            f"Context Precision     : "
            f"{summary['context_precision']:.3f}"
        )

        print(
            f"Context Recall        : "
            f"{summary['context_recall']:.3f}"
        )

        print(
            f"Average Latency       : "
            f"{summary['latency']:.2f}s"
        )

        print(
            f"Planner Iterations    : "
            f"{summary['planner_iterations']:.2f}"
        )

        print(
            f"Tool Calls            : "
            f"{summary['tool_calls']:.2f}"
        )

        print(
            f"Citations             : "
            f"{summary['citation_count']:.2f}"
        )

        print(
            f"Retrieved Documents   : "
            f"{summary['retrieved_documents']:.2f}"
        )

        print(
            f"Retriever Usage       : "
            f"{summary['retriever_usage_rate'] * 100:.1f}%"
        )

        print()

        print("=" * 60)

    # ==========================================================
    # Markdown Report
    # ==========================================================

    def save_markdown(
        self,
        results: list[EvaluationResult],
    ):

        summary = self._summary(
            results
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        path = (
            self.output_dir
            / f"{timestamp}.md"
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                "# Enterprise Evaluation Report\n\n"
            )

            f.write(
                "## Summary\n\n"
            )

            for key, value in summary.items():

                if isinstance(
                    value,
                    float,
                ):

                    f.write(
                        f"- **{key}** : "
                        f"{value:.4f}\n"
                    )

                else:

                    f.write(
                        f"- **{key}** : "
                        f"{value}\n"
                    )

            f.write(
                "\n## Individual Results\n\n"
            )

            for index, result in enumerate(
                results,
                start=1,
            ):

                f.write(
                    f"### Question {index}\n\n"
                )

                f.write(
                    f"**Question:** "
                    f"{result.sample.question}\n\n"
                )

                f.write(
                    f"**Answer:** "
                    f"{result.sample.answer}\n\n"
                )

                f.write(
                    "**Metrics:**\n\n"
                )

                for key, value in vars(
                    result.metrics
                ).items():

                    f.write(
                        f"- **{key}** : "
                        f"{value}\n"
                    )

                f.write("\n---\n\n")

        logger.info(
            "Markdown report saved to %s",
            path,
        )

    # ==========================================================
    # JSON Report
    # ==========================================================

    def save_json(
        self,
        results: list[EvaluationResult],
    ):

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        path = (
            self.output_dir
            / f"{timestamp}.json"
        )

        output = []

        for result in results:

            output.append(

                {
                    "question":
                        result.sample.question,

                    "answer":
                        result.sample.answer,

                    "ground_truth":
                        result.sample.ground_truth,

                    "contexts":
                        result.sample.contexts,

                    "citations":
                        result.sample.citations,

                    "latency":
                        result.sample.latency,

                    "metadata":
                        result.sample.metadata,

                    "metrics":
                        vars(
                            result.metrics
                        ),
                }

            )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                output,
                f,
                indent=4,
                default=str,
            )

        logger.info(
            "JSON report saved to %s",
            path,
        )