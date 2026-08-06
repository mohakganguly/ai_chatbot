"""
reranker.py

Enterprise Reranker.

Responsible for:
1. Reranking retrieved documents
2. Selecting the most relevant chunks
"""

from __future__ import annotations

import time
from typing import List

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from config import (
    RERANKER_MODEL,
    RERANK_TOP_K,
)

from utils.logger import get_logger

logger = get_logger(__name__)


class Reranker:

    def __init__(self):

        logger.info(
            "Initializing Reranker '%s'",
            RERANKER_MODEL,
        )
        print("Before CrossEncoder")

        try:
            self.model = CrossEncoder(RERANKER_MODEL)
            print("After CrossEncoder")

        except Exception as e:
            print("CrossEncoder Exception:")
            print(type(e))
            print(e)
            raise

        finally:
            print("Finally block reached")
        logger.info(
            "Reranker initialized successfully."
        )

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = RERANK_TOP_K,
    ) -> List[Document]:

        logger.info(
            "Reranking %d document(s).",
            len(documents),
        )

        if not documents:
            return []

        start = time.perf_counter()

        try:

            pairs = [
                (query, doc.page_content)
                for doc in documents
            ]

            scores = self.model.predict(
                pairs
            )

            ranked = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True,
            )

            reranked_documents = []

            for document, score in ranked[:top_k]:
                document.metadata["reranker_score"] = float(score)

                reranked_documents.append(document)

            elapsed = (
                time.perf_counter() - start
            )

            logger.info(
                "Selected top-%d document(s).",
                len(reranked_documents),
            )

            logger.info(
                "Reranking completed in %.3f sec.",
                elapsed,
            )

            return reranked_documents

        except Exception:

            logger.exception(
                "Reranking failed."
            )

            logger.info(
                "Returning original retrieval results."
            )

            return documents[:top_k]

if __name__ == "__main__":

    from rag.retrieval.retriever import Retriever

    retriever = Retriever()

    reranker = Reranker()

    query = "Explain Git Merge"

    docs = retriever.retrieve(query)

    docs = reranker.rerank(
        query=query,
        documents=docs,
    )

    print("=" * 80)

    for i, doc in enumerate(docs, start=1):

        print(f"\nDocument {i}")

        print(doc.page_content[:500])

        print("-" * 80)