"""
retriever.py

Enterprise Hybrid Retriever.

Retrieval pipeline:

Query
  ├── Dense embedding
  └── BM25 text
          ↓
   Qdrant Hybrid Search
          ↓
         RRF
          ↓
    Hybrid candidates
"""

from __future__ import annotations

import time
from typing import List

from langchain_core.documents import Document

from config import HYBRID_TOP_K

from rag.embedding.embedding_model import (
    get_embedding_model,
)

from rag.vectorstore.vector_store import (
    QdrantVectorStore,
)

from utils.logger import get_logger


logger = get_logger(__name__)


class Retriever:
    """
    Hybrid retriever using:

    Dense semantic retrieval
    +
    BM25 lexical retrieval
    +
    RRF fusion
    """

    def __init__(self,collection_name:str|None=None):

        logger.info(
            "Initializing Hybrid Retriever"
        )

        self.embedder = get_embedding_model()

        self.vector_store = QdrantVectorStore(collection_name=collection_name)

        logger.info(
            "Hybrid Retriever initialized successfully."
        )

    def retrieve(
        self,
        query: str,
        thread_id: str,
        top_k: int = HYBRID_TOP_K,
    ) -> List[Document]:

        logger.info(
            "Retrieving top-%d hybrid document(s).",
            top_k,
        )

        logger.info(
            "Query: %s",
            query,
        )

        start = time.perf_counter()

        try:

            # ==================================================
            # Dense Query Embedding
            # ==================================================

            embedding_start = time.perf_counter()

            query_vector = (
                self.embedder.embed_query(query)
            )

            embedding_time = (
                time.perf_counter()
                - embedding_start
            )

            logger.info(
                "Query embedding generated in %.3f sec.",
                embedding_time,
            )

            # ==================================================
            # Hybrid Qdrant Search
            #
            # Qdrant receives:
            #
            # 1. query text       → BM25
            # 2. query vector     → Dense
            #
            # Vector store performs:
            #
            # Dense Top-K
            #       +
            # BM25 Top-K
            #       ↓
            #      RRF
            #       ↓
            # Hybrid Top-K
            # ==================================================

            search_start = time.perf_counter()

            documents = self.vector_store.search(
                query=query,
                query_vector=query_vector,
                limit=top_k,
                thread_id=thread_id,
            )

            search_time = (
                time.perf_counter()
                - search_start
            )

            logger.info(
                "Hybrid Qdrant search completed in %.3f sec.",
                search_time,
            )

            # ==================================================
            # Total Retrieval Time
            # ==================================================

            elapsed = (
                time.perf_counter()
                - start
            )

            logger.info(
                "Hybrid retrieval completed in %.3f sec.",
                elapsed,
            )

            logger.info(
                "Retrieved %d document(s).",
                len(documents),
            )

            return documents

        except Exception:

            logger.exception(
                "Hybrid retrieval failed."
            )

            raise


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    retriever = Retriever()

    query = "What is a Zombie Process?"

    results = retriever.retrieve(
        query=query,
        thread_id=None,
    )

    print("=" * 80)
    print(f"Retrieved {len(results)} documents")

    for i, document in enumerate(results, start=1):

        print(f"\nResult {i}")

        print("Metadata:")
        print(document.metadata)

        print("\nContent:")
        print(document.page_content[:500])

        print("-" * 80)