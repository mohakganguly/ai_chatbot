"""
context_filter.py

Enterprise Context Filter.

Responsible for:

1. Removing low relevance chunks
2. Removing duplicate chunks
3. Enforcing context size limits
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from config import (
    MIN_RERANK_SCORE,
    MAX_CONTEXT_DOCUMENTS,
    MAX_CONTEXT_CHARACTERS,
)

from utils.logger import get_logger


logger = get_logger(__name__)


class ContextFilter:
    """
    Filters reranked documents before they are
    passed to the answer generation stage.
    """

    def __init__(self):

        logger.info(
            "Initializing Context Filter"
        )

    # ======================================================
    # Main Filtering Pipeline
    # ======================================================

    def filter_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:

        logger.info(
            "Filtering %d retrieved document(s).",
            len(documents),
        )

        if not documents:

            logger.info(
                "No documents available for filtering."
            )

            return []

        # --------------------------------------------------
        # Score filtering
        # --------------------------------------------------

        # Disabled for now.
        #
        # Cross-encoder scores are not being treated as
        # calibrated relevance probabilities. We should
        # determine a threshold empirically before enabling
        # this filter.
        #
        # documents = self._filter_by_score(documents)

        # --------------------------------------------------
        # Remove exact duplicate chunks
        # --------------------------------------------------

        documents = self._remove_duplicates(
            documents
        )

        # --------------------------------------------------
        # Apply context budget
        # --------------------------------------------------

        documents = self._apply_context_budget(
            documents
        )

        logger.info(
            "Final Context Size : %d document(s).",
            len(documents),
        )

        return documents

    # ======================================================
    # Filter by Reranker Score
    # ======================================================

    def _filter_by_score(
        self,
        documents: List[Document],
    ) -> List[Document]:

        filtered = []

        removed = 0

        for document in documents:

            score = document.metadata.get(
                "reranker_score",
                0.0,
            )

            if score >= MIN_RERANK_SCORE:

                filtered.append(document)

            else:

                removed += 1

        logger.info(
            "Removed %d low-score chunk(s).",
            removed,
        )

        return filtered

    # ======================================================
    # Remove Exact Duplicate Chunks
    # ======================================================

    def _remove_duplicates(
        self,
        documents: List[Document],
    ) -> List[Document]:

        unique_documents = []

        seen = set()

        removed = 0

        for document in documents:

            normalized = (
                document.page_content
                .strip()
                .lower()
            )

            if not normalized:

                removed += 1

                continue

            if normalized in seen:

                removed += 1

                continue

            seen.add(normalized)

            unique_documents.append(
                document
            )

        logger.info(
            "Removed %d duplicate/empty chunk(s).",
            removed,
        )

        return unique_documents

    # ======================================================
    # Context Budget
    # ======================================================

    def _apply_context_budget(
        self,
        documents: List[Document],
    ) -> List[Document]:

        selected = []

        total_characters = 0

        for document in documents:

            # Maximum number of documents
            if (
                len(selected)
                >= MAX_CONTEXT_DOCUMENTS
            ):

                break

            length = len(
                document.page_content
            )

            # Skip this document if it doesn't fit.
            #
            # IMPORTANT:
            # Use continue instead of break so that a
            # smaller, highly-ranked document later in
            # the list can still fit into the budget.
            if (
                total_characters + length
                > MAX_CONTEXT_CHARACTERS
            ):

                continue

            selected.append(
                document
            )

            total_characters += length

        logger.info(
            "Context Budget : %d characters.",
            total_characters,
        )

        logger.info(
            "Selected %d document(s) "
            "within context budget.",
            len(selected),
        )

        return selected


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    from rag.retrieval.retriever import Retriever
    from rag.retrieval.reranker import Reranker

    logger.info(
        "Starting Context Filter test."
    )

    retriever = Retriever()

    reranker = Reranker()

    context_filter = ContextFilter()

    query = (
        "Important interview questions "
        "from Git"
    )

    logger.info(
        "Query : %s",
        query,
    )

    # ------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------

    documents = retriever.retrieve(
        query=query,
        thread_id="test-thread",
        top_k=8,
    )

    logger.info(
        "Retrieved %d document(s).",
        len(documents),
    )

    # ------------------------------------------------------
    # Reranking
    # ------------------------------------------------------

    documents = reranker.rerank(
        query=query,
        documents=documents,
        top_k=5,
    )

    logger.info(
        "Reranked %d document(s).",
        len(documents),
    )

    # ------------------------------------------------------
    # Context Filtering
    # ------------------------------------------------------

    documents = context_filter.filter_documents(
        documents
    )

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    print("=" * 80)

    print(
        "Final Documents:",
        len(documents),
    )

    for index, document in enumerate(
        documents,
        start=1,
    ):

        print(
            f"\nDocument {index}"
        )

        print(
            "Reranker Score:",
            document.metadata.get(
                "reranker_score"
            ),
        )

        print(
            document.page_content[:500]
        )

        print("-" * 80)

    logger.info(
        "Context Filter test completed."
    )