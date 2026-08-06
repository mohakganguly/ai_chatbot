"""
run.py

Runs enterprise evaluation benchmarks.
"""
from __future__ import annotations

print("RUN START")

import argparse
import json
from pathlib import Path

from evals.evaluator import Evaluator
from evals.report import ReportGenerator

from utils.logger import get_logger
print("IMPORTS COMPLETE")
logger = get_logger(__name__)


def load_suite(
    suite: str,
):

    benchmark_dir = Path(
        "evals/benchmarks"
    )

    file = benchmark_dir / f"{suite}.json"

    if not file.exists():

        raise FileNotFoundError(
            file
        )

    with open(

        file,

        "r",

        encoding="utf-8",

    ) as f:

        return json.load(f)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--suite",

        default="rag",

        help="Benchmark suite",

    )

    args = parser.parse_args()

    logger.info(

        "Loading benchmark suite: %s",

        args.suite,

    )

    benchmark = load_suite(
        args.suite
    )

    evaluator = Evaluator()

    report = ReportGenerator()

    results = []

    total = len(
        benchmark
    )

    for i, sample in enumerate(
        benchmark,
        start=1,
    ):

        logger.info(

            "[%d/%d] %s",

            i,

            total,

            sample["question"],

        )

        result = evaluator.evaluate(

            question=sample["question"],

            ground_truth=sample.get(
                "ground_truth"
            ),

        )

        results.append(
            result
        )

    report.print_report(
        results
    )

    report.save_markdown(
        results
    )

    report.save_json(
        results
    )

    logger.info(
        "Evaluation Completed"
    )


if __name__ == "__main__":

    main()