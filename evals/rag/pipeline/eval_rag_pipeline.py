"""
Pipeline-level RAG Evaluation.

Production pipeline evaluated:

User Query
    ↓
RetrievalService
    ↓
Query Rewrite / Retrieval
    ↓
Reranking
    ↓
Context Filtering
    ↓
PromptBuilder
    ↓
Production answer_llm
    ↓
Generated Answer
    ↓
DeepEval RAG Triad


RAG Triad:

1. Contextual Relevancy
2. Faithfulness
3. Answer Relevancy
"""

from __future__ import annotations

import json

from pathlib import Path


# ==========================================================
# DeepEval
# ==========================================================

from deepeval import evaluate

from deepeval.metrics import (
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)

from deepeval.test_case import (
    LLMTestCase,
)


# ==========================================================
# LangChain
# ==========================================================

from langchain_core.messages import (
    HumanMessage,
)


# ==========================================================
# Production Components
# ==========================================================

from core.llm import (
    answer_llm,
)

from rag.services.retrieval_service import (
    get_retrieval_service,
)

from rag.prompts.prompt_builder import (
    PromptBuilder,
)


# ==========================================================
# Evaluation Model
# ==========================================================

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
    .parent
)


DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "generation_eval.json"
)


# ==========================================================
# Production Services
# ==========================================================

retrieval_service = (
    get_retrieval_service()
)


prompt_builder = (
    PromptBuilder()
)


# ==========================================================
# Evaluation Model
# ==========================================================

evaluation_model = (
    OpenRouterDeepEvalModel()
)


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset() -> list[dict]:
    """
    Load the pipeline evaluation dataset.

    Expected format:

    [
        {
            "input": "...",
            "expected_output": "..."
        }
    ]
    """

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ==========================================================
# Run Complete Production RAG Pipeline
# ==========================================================

def run_rag_pipeline(
    query: str,
    test_case_id: int,
) -> dict:
    """
    Run the complete production RAG pipeline.

    Pipeline:

    Query
        ↓
    RetrievalService
        ↓
    Retrieved Documents
        ↓
    PromptBuilder
        ↓
    answer_llm
        ↓
    Generated Answer
    """

    # ------------------------------------------------------
    # Step 1: Retrieval
    # ------------------------------------------------------

    print(
        "\nRunning Retrieval Pipeline..."
    )

    retrieval_result = (
        retrieval_service.retrieve(
            query=query,

            thread_id=(
                f"pipeline-eval-{test_case_id}"
            ),

            messages=[
                HumanMessage(
                    content=query
                )
            ],
        )
    )


    documents = (
        retrieval_result.documents
    )


    print(
        f"Retrieved {len(documents)} document(s)."
    )


    # ------------------------------------------------------
    # Step 2: Convert Documents to Context
    # ------------------------------------------------------

    context_chunks = [
        document.page_content
        for document in documents
    ]


    # ------------------------------------------------------
    # Step 3: Build Production Prompt
    # ------------------------------------------------------

    print(
        "\nBuilding Prompt..."
    )

    prompt = (
        prompt_builder.build(
            query=query,
            documents=documents,
        )
    )


    # ------------------------------------------------------
    # Step 4: Generate Answer
    # ------------------------------------------------------

    print(
        "\nGenerating Answer..."
    )

    response = (
        answer_llm.invoke(
            prompt
        )
    )


    generated_answer = (
        response.content
    )


    # ------------------------------------------------------
    # Return Pipeline Result
    # ------------------------------------------------------

    return {

        "answer": generated_answer,

        "context": context_chunks,

        "documents": documents,

        "rewritten_query": (
            retrieval_result.rewritten_query
        ),

        "prompt": prompt,

    }


# ==========================================================
# Build DeepEval Test Cases
# ==========================================================

def build_test_cases() -> list[LLMTestCase]:
    """
    Run the complete production RAG pipeline
    for every evaluation dataset item.

    The retrieval context is obtained dynamically
    from the real production retriever.
    """

    dataset = (
        load_dataset()
    )


    test_cases = []


    for index, item in enumerate(
        dataset,
        start=1,
    ):

        # --------------------------------------------------
        # Dataset Values
        # --------------------------------------------------

        query = (
            item["input"]
        )


        expected_output = (
            item.get(
                "expected_output"
            )
        )


        # --------------------------------------------------
        # Print Test Case
        # --------------------------------------------------

        print(
            "\n"
            + "=" * 80
        )


        print(
            f"PIPELINE TEST CASE #{index}"
        )


        print(
            "=" * 80
        )


        print(
            f"\nQuery:\n{query}"
        )


        # --------------------------------------------------
        # Run Production Pipeline
        # --------------------------------------------------

        pipeline_result = (
            run_rag_pipeline(
                query=query,
                test_case_id=index,
            )
        )


        generated_answer = (
            pipeline_result[
                "answer"
            ]
        )


        context_chunks = (
            pipeline_result[
                "context"
            ]
        )


        rewritten_query = (
            pipeline_result[
                "rewritten_query"
            ]
        )


        # --------------------------------------------------
        # Display Results
        # --------------------------------------------------

        print(
            "\n"
            + "-" * 80
        )


        print(
            "\nOriginal Query:"
        )

        print(
            query
        )


        print(
            "\nRetrieval Query:"
        )

        print(
            rewritten_query
        )


        print(
            "\nNumber of Retrieved Context Chunks:"
        )

        print(
            len(context_chunks)
        )


        # --------------------------------------------------
        # Display Retrieved Context
        # --------------------------------------------------

        for context_index, context in enumerate(
            context_chunks,
            start=1,
        ):

            print(
                "\n"
                + "-" * 80
            )

            print(
                f"Context Chunk #{context_index}"
            )

            print(
                "-" * 80
            )

            print(
                context
            )


        # --------------------------------------------------
        # Display Generated Answer
        # --------------------------------------------------

        print(
            "\n"
            + "-" * 80
        )


        print(
            "Generated Answer:"
        )


        print(
            "-" * 80
        )


        print(
            generated_answer
        )


        # --------------------------------------------------
        # Create DeepEval Test Case
        # --------------------------------------------------

        test_case = (
            LLMTestCase(

                input=query,

                actual_output=(
                    generated_answer
                ),

                retrieval_context=(
                    context_chunks
                ),

                expected_output=(
                    expected_output
                ),

            )
        )


        test_cases.append(
            test_case
        )


    return test_cases


# ==========================================================
# Run Evaluation
# ==========================================================

def run_evaluation() -> None:
    """
    Run pipeline-level RAG evaluation
    using the RAG Triad.
    """

    # ------------------------------------------------------
    # Build Test Cases
    # ------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )


    print(
        "BUILDING PIPELINE TEST CASES"
    )


    print(
        "=" * 80
    )


    test_cases = (
        build_test_cases()
    )


    # ------------------------------------------------------
    # Evaluation Start
    # ------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )


    print(
        "RUNNING RAG TRIAD EVALUATION"
    )


    print(
        "=" * 80
    )


    # ======================================================
    # RAG TRIAD METRIC 1
    # Context Relevancy
    # ======================================================

    contextual_relevancy = (
        ContextualRelevancyMetric(

            model=(
                evaluation_model
            ),

            threshold=0.7,

        )
    )


    # ======================================================
    # RAG TRIAD METRIC 2
    # Faithfulness
    # ======================================================

    faithfulness = (
        FaithfulnessMetric(

            model=(
                evaluation_model
            ),

            threshold=0.7,

        )
    )


    # ======================================================
    # RAG TRIAD METRIC 3
    # Answer Relevancy
    # ======================================================

    answer_relevancy = (
        AnswerRelevancyMetric(

            model=(
                evaluation_model
            ),

            threshold=0.7,

        )
    )


    # ======================================================
    # Metrics
    # ======================================================

    metrics = [

        contextual_relevancy,

        faithfulness,

        answer_relevancy,

    ]


    # ======================================================
    # Run DeepEval
    # ======================================================

    evaluate(

        test_cases=test_cases,

        metrics=metrics,

    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    run_evaluation()