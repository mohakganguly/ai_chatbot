"""
retriever.py

Enterprise Retriever.

Responsible for:
1. Embedding the query
2. Searching the vector database
3. Returning relevant documents
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from config import RERANK_TOP_K

from rag.embedding.embedding_model import get_embedding_model
from rag.vectorstore.vector_store import QdrantVectorStore

from utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:

    def __init__(self):

        logger.info("Initializing Retriever")

        self.embedder = get_embedding_model()

        self.vector_store = QdrantVectorStore()

    def retrieve(
        self,
        query: str,
        thread_id: str,
        top_k: int = RERANK_TOP_K
    ) -> List[Document]:

        logger.info(
            "Retrieving top-%d document(s).",
            top_k,
        )

        logger.info("Query : %s", query)

        query_vector = self.embedder.embed_query(query)

        logger.info("Generated query embedding")

        documents = self.vector_store.search(
            query_vector=query_vector,
            limit=top_k,
            thread_id=thread_id,
        )

        logger.info(
            "Retrieved %d document(s).",
            len(documents),
        )

        return documents



if __name__ == "__main__":

    retriever = Retriever()

    results = retriever.retrieve(
        "What is a Zombie Process?"
    )

    print("=" * 80)

    for i, document in enumerate(results, start=1):

        print(f"\nResult {i}")

        print(document.metadata)

        print()

        print(document.page_content[:500])

        print("-" * 80)