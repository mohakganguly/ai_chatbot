"""
reranker_benchmark.py

Benchmarks reranker latency on the actual RAG retrieval results.

Models:
1. BAAI/bge-reranker-base
2. cross-encoder/ms-marco-MiniLM-L-6-v2
"""

from __future__ import annotations

import time
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from rag.retrieval.retriever import Retriever
from utils.logger import get_logger


logger = get_logger(__name__)


MODELS = [
    "BAAI/bge-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
]


def benchmark_model(
    model_name: str,
    query: str,
    documents: List[Document],
    top_k: int = 5,
):
    logger.info("=" * 80)
    logger.info("Testing reranker: %s", model_name)
    logger.info("=" * 80)

    # ------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------

    start = time.perf_counter()

    model = CrossEncoder(
        model_name,
        max_length=256,
    )

    initialization_time = (
        time.perf_counter() - start
    )

    logger.info(
        "Model initialization: %.3f sec",
        initialization_time,
    )

    # ------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------

    warmup_pairs = [
        (
            query,
            documents[0].page_content,
        )
    ]

    model.predict(
        warmup_pairs,
        batch_size=1,
        show_progress_bar=False,
    )

    # ------------------------------------------------------
    # Actual benchmark
    # ------------------------------------------------------

    pairs = [
        (
            query,
            document.page_content,
        )
        for document in documents
    ]

    start = time.perf_counter()

    scores = model.predict(
        pairs,
        batch_size=8,
        show_progress_bar=False,
    )

    inference_time = (
        time.perf_counter() - start
    )

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    results = ranked[:top_k]

    logger.info(
        "Inference time for %d candidates: %.3f sec",
        len(documents),
        inference_time,
    )

    logger.info(
        "Total cold-start time: %.3f sec",
        initialization_time + inference_time,
    )

    logger.info(
        "Top-%d results:",
        len(results),
    )

    for index, (document, score) in enumerate(
        results,
        start=1,
    ):

        logger.info(
            "%d. score=%.4f | %s",
            index,
            float(score),
            document.page_content[:120].replace(
                "\n",
                " ",
            ),
        )

    return {
        "model": model_name,
        "initialization": initialization_time,
        "inference": inference_time,
        "total": initialization_time + inference_time,
    }


def main():

    query = "What is a Zombie Process?"

    # ------------------------------------------------------
    # Retrieve candidates ONCE
    # ------------------------------------------------------

    logger.info("=" * 80)
    logger.info("Retrieving hybrid candidates")
    logger.info("=" * 80)

    retriever = Retriever()

    documents = retriever.retrieve(
        query=query,
        thread_id=None,
        top_k=10,
    )

    logger.info(
        "Retrieved %d candidates.",
        len(documents),
    )

    if not documents:

        logger.error(
            "No documents retrieved."
        )

        return

    # ------------------------------------------------------
    # Benchmark both models
    # ------------------------------------------------------

    results = []

    for model_name in MODELS:

        result = benchmark_model(
            model_name=model_name,
            query=query,
            documents=documents,
            top_k=5,
        )

        results.append(result)

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    logger.info("=" * 80)
    logger.info("RERANKER BENCHMARK")
    logger.info("=" * 80)

    for result in results:

        logger.info(
            "%s",
            result["model"],
        )

        logger.info(
            "Initialization : %.3f sec",
            result["initialization"],
        )

        logger.info(
            "Inference      : %.3f sec",
            result["inference"],
        )

        logger.info(
            "Total          : %.3f sec",
            result["total"],
        )

        logger.info("-" * 80)


if __name__ == "__main__":
    main()