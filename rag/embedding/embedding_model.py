"""
embedding_model.py

Central embedding service used throughout the RAG pipeline.
"""

from __future__ import annotations

import time
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    NORMALIZE_EMBEDDINGS,
)
from rag.models.embedded_chunk import EmbeddedChunk
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """
    Central embedding service used throughout the RAG pipeline.
    """

    def __init__(self):

        logger.info(
            "Initializing embedding model '%s' on device '%s'.",
            EMBEDDING_MODEL,
            EMBEDDING_DEVICE,
        )

        try:

            self.embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={
                    "device": EMBEDDING_DEVICE,
                },
                encode_kwargs={
                    "normalize_embeddings": NORMALIZE_EMBEDDINGS,
                },
            )

            logger.info(
                "Embedding model initialized successfully."
            )

        except Exception:
            logger.exception(
                "Failed to initialize embedding model."
            )
            raise

    def embed_documents(
        self,
        documents: List[Document],
    ) -> List[EmbeddedChunk]:

        logger.info(
            "Generating embeddings for %d document(s).",
            len(documents),
        )

        start = time.perf_counter()

        try:

            texts = [
                document.page_content
                for document in documents
            ]

            vectors = self.embedding_model.embed_documents(
                texts
            )

            embedded_chunks = []

            for document, vector in zip(documents, vectors):

                embedded_chunks.append(
                    EmbeddedChunk(
                        document=document,
                        vector=vector,
                    )
                )

            elapsed = time.perf_counter() - start

            logger.info(
                "Generated %d embedding(s) in %.3f sec.",
                len(embedded_chunks),
                elapsed,
            )

            if embedded_chunks:

                logger.info(
                    "Embedding dimension: %d",
                    len(embedded_chunks[0].vector),
                )

            return embedded_chunks

        except Exception:
            logger.exception(
                "Failed while generating document embeddings."
            )
            raise

    def embed_query(
        self,
        query: str,
    ) -> List[float]:

        logger.info(
            "Generating query embedding."
        )

        start = time.perf_counter()

        try:

            vector = self.embedding_model.embed_query(
                query
            )

            elapsed = time.perf_counter() - start

            logger.info(
                "Query embedding generated in %.3f sec.",
                elapsed,
            )

            return vector

        except Exception:
            logger.exception(
                "Failed while generating query embedding."
            )
            raise


_embedder = EmbeddingModel()


def get_embedding_model() -> EmbeddingModel:

    return _embedder


# ---------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------

if __name__ == "__main__":

    docs = [
        Document(page_content="Linux is an operating system."),
        Document(page_content="Messi won the FIFA World Cup."),
    ]

    logger.info("Starting embedding model test.")

    embedder = get_embedding_model()

    start = time.perf_counter()

    embedded_chunks = embedder.embed_documents(docs)

    elapsed = time.perf_counter() - start

    logger.info(
        "Embedding test completed in %.3f sec.",
        elapsed,
    )

    logger.info(
        "Number of embeddings: %d",
        len(embedded_chunks),
    )

    if embedded_chunks:

        logger.info(
            "Embedding dimension: %d",
            len(embedded_chunks[0].vector),
        )

        logger.info(
            "First 10 values: %s",
            embedded_chunks[0].vector[:10],
        )

    logger.info("Embedding model test completed successfully.")