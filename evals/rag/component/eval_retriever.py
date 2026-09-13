"""
Component-level evaluation for the Retriever.

Pipeline evaluated:

Query
  ↓
Retriever
  ↓
Top-K Documents
  ↓
DeepEval
"""

#recall improvement -> change chunk_size,chunk_overlap
#precision improvement -> reranking


from __future__ import annotations

import json
import uuid
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

from rag.retrieval.retriever import Retriever

from config import EVAL_COLLECTION_NAME
from evals.models.openrouter_model import (
    OpenRouterDeepEvalModel,
)
# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "retrieval_eval.json"
)


# ==========================================================
# Retriever
# ==========================================================

retriever = Retriever(collection_name=EVAL_COLLECTION_NAME)


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset() -> list[dict]:
    """
    Load the retrieval evaluation dataset.
    """

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ==========================================================
# Retrieve Context
# ==========================================================

def retrieve_context(
    query: str,
    top_k: int = 8,
) -> list[str]:
    """
    Run only the raw Retriever.

    No query rewriting.
    No reranking.
    No context filtering.
    """

    documents = retriever.retrieve(
        query=query,
        thread_id=None,
        top_k=top_k,
    )

    return [
        document.page_content
        for document in documents
    ]


# ==========================================================
# Build Test Cases
# ==========================================================

def build_test_cases() -> list[LLMTestCase]:
    """
    Run the retriever for every evaluation query.
    """

    dataset = load_dataset()

    test_cases = []

    for item in dataset:

        query = item["input"]

        expected_output = item[
            "expected_output"
        ]

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"Query: {query}"
        )

        retrieval_context = (
            retrieve_context(
                query=query,
                top_k=8,
            )
        )

        print(
            f"Retrieved documents: "
            f"{len(retrieval_context)}"
        )

        test_case = LLMTestCase(

            input=query,

            actual_output="",

            expected_output=expected_output,

            retrieval_context=retrieval_context,
        )

        test_cases.append(
            test_case
        )

    return test_cases


# ==========================================================
# Evaluation
# ==========================================================

def run_evaluation() -> None:

    test_cases = build_test_cases()

    evaluation_model = OpenRouterDeepEvalModel(
        max_concurrent=3,
        max_retries=20,
    )

    metrics = [
        ContextualPrecisionMetric(
            model=evaluation_model,
        ),
        ContextualRecallMetric(
            model=evaluation_model,
        ),
        ContextualRelevancyMetric(
            model=evaluation_model,
        ),
    ]

    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    run_evaluation()