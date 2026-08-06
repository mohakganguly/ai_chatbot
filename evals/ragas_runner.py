"""
ragas_runner.py

Enterprise RAGAS runner.

Responsible for converting EvaluationSamples
into a RAGAS EvaluationDataset and executing
the evaluation.
"""

from __future__ import annotations

from typing import List

from ragas import evaluate

from ragas.dataset_schema import (
    EvaluationDataset,
    SingleTurnSample,
)

from evals.schemas import (
    EvaluationSample,
    EvaluationResult,
)

from evals.metrics import EvaluationMetrics

from evals.config import (
    RAGAS_LLM,
    RAGAS_EMBEDDINGS,
    RAGAS_METRICS,
)

from utils.logger import get_logger

logger = get_logger(__name__)


class RagasRunner:
    """
    Executes RAGAS evaluation over one
    or more EvaluationSamples.
    """

    def __init__(self):

        self.metrics = RAGAS_METRICS

        self.llm = RAGAS_LLM

        self.embeddings = RAGAS_EMBEDDINGS

    # =====================================================
    # EvaluationSample -> SingleTurnSample
    # =====================================================

    @staticmethod
    def _convert_sample(
        sample: EvaluationSample,
    ) -> SingleTurnSample:

        return SingleTurnSample(

            user_input=sample.question,

            response=sample.answer,

            retrieved_contexts=sample.contexts,

            reference=sample.ground_truth,
        )

    # =====================================================
    # Build EvaluationDataset
    # =====================================================

    def _build_dataset(

        self,

        samples: List[EvaluationSample],

    ) -> EvaluationDataset:

        logger.info(

            "Building RAGAS Evaluation Dataset"

        )

        ragas_samples = [

            self._convert_sample(sample)

            for sample in samples

        ]

        dataset = EvaluationDataset(

            samples=ragas_samples

        )

        logger.info(

            "Dataset Size : %d",

            len(ragas_samples),

        )

        return dataset

    # =====================================================
    # Execute RAGAS Evaluation
    # =====================================================

    def evaluate(
        self,
        samples: List[EvaluationSample],
    ) -> List[EvaluationResult]:

        logger.info(
            "Starting RAGAS Evaluation"
        )

        dataset = self._build_dataset(
            samples
        )

        try:

            ragas_result = evaluate(

                dataset=dataset,

                metrics=self.metrics,

                llm=self.llm,

                embeddings=self.embeddings,

                show_progress=True,

                raise_exceptions=True,

            )

        except Exception:

            logger.exception(
                "RAGAS Evaluation Failed"
            )

            raise

        logger.info(
            "RAGAS Evaluation Finished"
        )

        scores = ragas_result.to_pandas()

        logger.info(
            "Evaluated %d samples",
            len(scores),
        )

        results: List[
            EvaluationResult
        ] = []

        # ===============================================
        # Convert RAGAS Results
        # ===============================================

        for sample, row in zip(
            samples,
            scores.to_dict(
                orient="records"
            ),
        ):

            metrics = EvaluationMetrics(

                faithfulness=row.get(
                    "faithfulness",
                    0.0,
                ),

                answer_relevancy=row.get(
                    "answer_relevancy",
                    0.0,
                ),

                context_precision=row.get(
                    "context_precision",
                    0.0,
                ),

                context_recall=row.get(
                    "context_recall",
                    0.0,
                ),

                latency=sample.latency,

                planner_iterations=sample.metadata.get(
                    "planner_iterations",
                    0,
                ),

                tool_calls=sample.metadata.get(
                    "tool_calls",
                    0,
                ),

                citations=len(
                    sample.citations
                ),

            )

            results.append(

                EvaluationResult(

                    sample=sample,

                    metrics=metrics,

                )

            )

        logger.info(
            "Generated %d EvaluationResults",
            len(results),
        )

        return results