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

    def __init__(self):

        logger.info(
            "Initializing Context Filter"
        )

    def filter_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:

        logger.info(
            "Filtering %d retrieved document(s).",
            len(documents),
        )

        # documents = self._filter_by_score(
        #     documents
        # )

        documents = self._remove_duplicates(
            documents
        )

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
    # Remove Duplicate Chunks
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

            if normalized in seen:

                removed += 1

                continue

            seen.add(normalized)

            unique_documents.append(document)

        logger.info(
            "Removed %d duplicate chunk(s).",
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

            if len(selected) >= MAX_CONTEXT_DOCUMENTS:
                break

            length = len(document.page_content)

            if (
                total_characters + length
                > MAX_CONTEXT_CHARACTERS
            ):
                break

            selected.append(document)

            total_characters += length

        logger.info(
            "Context Budget : %d characters",
            total_characters,
        )

        return selected

if __name__ == "__main__":

    from rag.retrieval.retriever import Retriever
    from rag.retrieval.reranker import Reranker

    retriever = Retriever()

    reranker = Reranker()

    context_filter = ContextFilter()

    query = "Important interview questions from Git"

    documents = retriever.retrieve(query)

    documents = reranker.rerank(
        query=query,
        documents=documents,
    )

    documents = context_filter.filter_documents(
        documents
    )

    print("=" * 80)

    print("Final Documents")

    print(len(documents))