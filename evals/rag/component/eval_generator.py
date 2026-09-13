"""
Component-level evaluation for the Generator.

Pipeline evaluated:

Fixed Context
      ↓
PromptBuilder
      ↓
Production answer_llm
      ↓
Generated Answer
      ↓
DeepEval
"""

#faithfullness, answer relevancy can only be improved by using better models and prompt tweaking

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from deepeval import evaluate

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)

from deepeval.metrics import GEval

from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
)

from core.llm import answer_llm

from rag.prompts.prompt_builder import (
    PromptBuilder,
)

from evals.models.openrouter_model import (
    OpenRouterDeepEvalModel,
)


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "generator_eval.json"
)


# ==========================================================
# Production Generator
# ==========================================================

prompt_builder = PromptBuilder()


# ==========================================================
# Evaluation Model
# ==========================================================

evaluation_model = OpenRouterDeepEvalModel()


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset() -> list[dict]:
    """
    Load the generator evaluation dataset.
    """

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ==========================================================
# Build Documents
# ==========================================================

def build_documents(
    context_chunks: list[str],
) -> list[Document]:
    """
    Convert dataset context strings
    into LangChain Documents.

    PromptBuilder expects:

    List[Document]
    """

    documents = []

    for index, chunk in enumerate(
        context_chunks,
        start=1,
    ):

        document = Document(
            page_content=chunk,
            metadata={
                "source": "evaluation_dataset",
                "page": index,
            },
        )

        documents.append(
            document
        )

    return documents


# ==========================================================
# Generate Answer
# ==========================================================

def generate_answer(
    query: str,
    context_chunks: list[str],
) -> str:
    """
    Run the exact production
    generation component.

    Pipeline:

    Context
        ↓
    PromptBuilder
        ↓
    answer_llm
        ↓
    Answer
    """

    documents = build_documents(
        context_chunks
    )

    prompt = prompt_builder.build(
        query=query,
        documents=documents,
    )

    response = answer_llm.invoke(
        prompt
    )

    return response.content


# ==========================================================
# Build Test Cases
# ==========================================================

def build_test_cases() -> list[LLMTestCase]:
    """
    Generate answers for every
    evaluation dataset item.
    """

    dataset = load_dataset()

    test_cases = []

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        query = item["input"]

        expected_output = item[
            "expected_output"
        ]

        context_chunks = item[
            "context"
        ]

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Test Case #{index}"
        )

        print(
            f"\nQuery:\n{query}"
        )

        print(
            "\nGenerating answer..."
        )

        generated_answer = (
            generate_answer(
                query=query,
                context_chunks=context_chunks,
            )
        )

        print(
            "\nGenerated Answer:"
        )

        print(
            generated_answer
        )

        print(
            "\nContext Chunks:"
        )

        print(
            len(context_chunks)
        )

        test_case = LLMTestCase(

            input=query,

            actual_output=generated_answer,

            expected_output=expected_output,

            retrieval_context=context_chunks,

        )

        test_cases.append(
            test_case
        )

    return test_cases


# ==========================================================
# Evaluation
# ==========================================================

def run_evaluation() -> None:
    """
    Run component-level
    Generator evaluation.
    """

    test_cases = build_test_cases()


    # ------------------------------------------------------
    # DeepEval Metrics
    # ------------------------------------------------------

    answer_relevancy = (
        AnswerRelevancyMetric(
            model=evaluation_model,
            threshold=0.7,
        )
    )


    faithfulness = (
        FaithfulnessMetric(
            model=evaluation_model,
            threshold=0.7,
        )
    )


    correctness = GEval(

        name="Answer Correctness",

        criteria=(
            "Evaluate whether the generated answer "
            "is factually correct and consistent with "
            "the expected answer."
        ),

        evaluation_params=[

            LLMTestCaseParams.INPUT,

            LLMTestCaseParams.ACTUAL_OUTPUT,

            LLMTestCaseParams.EXPECTED_OUTPUT,

        ],

        model=evaluation_model,

        threshold=0.7,

    )


    metrics = [

        answer_relevancy,

        faithfulness,

        correctness,

    ]


    # ------------------------------------------------------
    # Run DeepEval
    # ------------------------------------------------------

    evaluate(

        test_cases=test_cases,

        metrics=metrics,

    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    run_evaluation()