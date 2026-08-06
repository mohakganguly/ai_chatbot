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

        self.output_dir = Path("evals/reports")

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

    def _summary(

        self,

        results: list[EvaluationResult],

    ) -> dict:

        metrics = [

            result.metrics

            for result in results

        ]

        return {

            "questions": len(results),

            "faithfulness":

                self._average(

                    [

                        m.faithfulness

                        for m in metrics

                    ]

                ),

            "answer_relevancy":

                self._average(

                    [

                        m.answer_relevancy

                        for m in metrics

                    ]

                ),

            "context_precision":

                self._average(

                    [

                        m.context_precision

                        for m in metrics

                    ]

                ),

            "context_recall":

                self._average(

                    [

                        m.context_recall

                        for m in metrics

                    ]

                ),

            "latency":

                self._average(

                    [

                        m.latency

                        for m in metrics

                    ]

                ),

            "planner_iterations":

                self._average(

                    [

                        m.planner_iterations

                        for m in metrics

                    ]

                ),

            "tool_calls":

                self._average(

                    [

                        m.tool_calls

                        for m in metrics

                    ]

                ),

            "citation_count":

                self._average(

                    [

                        m.citation_count

                        for m in metrics

                    ]

                ),

        }

    # ==========================================================
    # Console Report
    # ==========================================================

    def print_report(

        self,

        results: list[EvaluationResult],

    ):

        summary = self._summary(results)

        print()

        print("=" * 60)

        print("Enterprise AI Evaluation Report")

        print("=" * 60)

        print()

        print(f"Questions             : {summary['questions']}")

        print(f"Faithfulness          : {summary['faithfulness']:.3f}")

        print(f"Answer Relevancy      : {summary['answer_relevancy']:.3f}")

        print(f"Context Precision     : {summary['context_precision']:.3f}")

        print(f"Context Recall        : {summary['context_recall']:.3f}")

        print(f"Average Latency       : {summary['latency']:.2f}s")

        print(f"Planner Iterations    : {summary['planner_iterations']:.2f}")

        print(f"Tool Calls            : {summary['tool_calls']:.2f}")

        print(f"Citations             : {summary['citation_count']:.2f}")

        print()

        print("=" * 60)

    # ==========================================================
    # Markdown
    # ==========================================================

    def save_markdown(

        self,

        results: list[EvaluationResult],

    ):

        summary = self._summary(results)

        timestamp = datetime.now().strftime(

            "%Y%m%d_%H%M%S"

        )

        path = self.output_dir / f"{timestamp}.md"

        with open(

            path,

            "w",

            encoding="utf-8",

        ) as f:

            f.write("# Enterprise Evaluation Report\n\n")

            for key, value in summary.items():

                f.write(f"- **{key}** : {value}\n")

        logger.info(

            "Markdown report saved to %s",

            path,

        )

    # ==========================================================
    # JSON
    # ==========================================================

    def save_json(

        self,

        results: list[EvaluationResult],

    ):

        timestamp = datetime.now().strftime(

            "%Y%m%d_%H%M%S"

        )

        path = self.output_dir / f"{timestamp}.json"

        output = []

        for result in results:

            output.append(

                {

                    "question":

                        result.sample.question,

                    "answer":

                        result.sample.answer,

                    "metrics":

                        vars(result.metrics),

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

            )

        logger.info(

            "JSON report saved to %s",

            path,

        )