"""
embedding_model.py

Central embedding service used throughout the RAG pipeline.

Features:
- Lazy model initialization
- Batch document embedding
- Query embedding
- Configurable device
- Configurable normalization
- Centralized singleton instance
"""

from __future__ import annotations

import time
from typing import List
import threading
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


# ==========================================================
# Embedding Model
# ==========================================================

class EmbeddingModel:
    """
    Central embedding service used throughout
    the RAG pipeline.

    The actual HuggingFace model is initialized
    lazily on the first embedding request.
    """

    def __init__(self):

        self.embedding_model = None
        self._model_lock = threading.Lock()
        logger.info(
            "Embedding service created. "
            "Model will be initialized lazily."
        )

    # ======================================================
    # Lazy Initialization
    # ======================================================

    def _initialize_model(self):

        if self.embedding_model is not None:
            return

        with self._model_lock:

            if self.embedding_model is not None:
                return self.embedding_model
            logger.info(
                "Initializing embedding model '%s' "
                "on device '%s'.",
                EMBEDDING_MODEL,
                EMBEDDING_DEVICE,
            )

            start = time.perf_counter()

            try:

                self.embedding_model = HuggingFaceEmbeddings(

                    model_name=EMBEDDING_MODEL,

                    model_kwargs={
                        "device": EMBEDDING_DEVICE,
                    },

                    encode_kwargs={
                        "normalize_embeddings":
                            NORMALIZE_EMBEDDINGS,
                    },
                )

                elapsed = (
                    time.perf_counter()
                    - start
                )

                logger.info(
                    "Embedding model initialized "
                    "successfully in %.3f sec.",
                    elapsed,
                )

            except Exception:

                logger.exception(
                    "Failed to initialize embedding model."
                )

                # Make sure a failed initialization
                # does not leave a partially initialized
                # model object behind.
                self.embedding_model = None

                raise

    # ======================================================
    # Document Embeddings
    # ======================================================

    def embed_documents(
        self,
        documents: List[Document],
    ) -> List[EmbeddedChunk]:

        if not documents:

            logger.info(
                "No documents supplied for embedding."
            )

            return []

        logger.info(
            "Generating embeddings for %d document(s).",
            len(documents),
        )

        start = time.perf_counter()

        try:

            # Initialize only when actually needed.
            self._initialize_model()

            texts = [
                document.page_content
                for document in documents
            ]

            # Batch embedding.
            vectors=self.embedding_model.embed_documents(texts)
            

            embedded_chunks = []

            for document, vector in zip(
                documents,
                vectors,
            ):

                embedded_chunks.append(
                    EmbeddedChunk(
                        document=document,
                        vector=vector,
                    )
                )

            elapsed = (
                time.perf_counter()
                - start
            )

            logger.info(
                "Generated %d embedding(s) "
                "in %.3f sec.",
                len(embedded_chunks),
                elapsed,
            )

            if embedded_chunks:

                logger.info(
                    "Embedding dimension: %d",
                    len(
                        embedded_chunks[0].vector
                    ),
                )

            return embedded_chunks

        except Exception:

            logger.exception(
                "Failed while generating "
                "document embeddings."
            )

            raise

    # ======================================================
    # Query Embedding
    # ======================================================

    def embed_query(
        self,
        query: str,
    ) -> List[float]:

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        logger.info(
            "Generating query embedding."
        )

        start = time.perf_counter()

        try:

            # Initialize only when actually needed.
            self._initialize_model()

            vector = (
                self.embedding_model.embed_query(
                    query
                )
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            logger.info(
                "Query embedding generated "
                "in %.3f sec.",
                elapsed,
            )

            return vector

        except Exception:

            logger.exception(
                "Failed while generating "
                "query embedding."
            )

            raise


# ==========================================================
# Singleton / Lazy Service
# ==========================================================

_embedder: EmbeddingModel | None = None
_embedder_lock = threading.Lock()


def get_embedding_model() -> EmbeddingModel:

    global _embedder

    if _embedder is not None:
        return _embedder

    with _embedder_lock:

        if _embedder is None:

            _embedder = EmbeddingModel()

    return _embedder


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    from langchain_core.documents import Document

    docs = [

        Document(
            page_content=(
                "Linux is an operating system."
            )
        ),

        Document(
            page_content=(
                "Messi won the FIFA World Cup."
            )
        ),
    ]

    logger.info(
        "Starting embedding model test."
    )

    # ------------------------------------------------------
    # Service creation should NOT load the model yet.
    # ------------------------------------------------------

    start = time.perf_counter()

    embedder = get_embedding_model()

    service_creation_time = (
        time.perf_counter()
        - start
    )

    logger.info(
        "Embedding service created in %.3f sec.",
        service_creation_time,
    )

    # ------------------------------------------------------
    # First embedding call
    # This is where lazy model initialization happens.
    # ------------------------------------------------------

    start = time.perf_counter()

    embedded_chunks = (
        embedder.embed_documents(
            docs
        )
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    logger.info(
        "First embedding call completed "
        "in %.3f sec.",
        elapsed,
    )

    logger.info(
        "Number of embeddings: %d",
        len(embedded_chunks),
    )

    if embedded_chunks:

        logger.info(
            "Embedding dimension: %d",
            len(
                embedded_chunks[0].vector
            ),
        )

        logger.info(
            "First 10 values: %s",
            embedded_chunks[0].vector[:10],
        )

    # ------------------------------------------------------
    # Second call
    # Model should already be loaded.
    # ------------------------------------------------------

    start = time.perf_counter()

    query_vector = embedder.embed_query(
        "What is an operating system?"
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    logger.info(
        "Query embedding completed "
        "in %.3f sec.",
        elapsed,
    )

    logger.info(
        "Query embedding dimension: %d",
        len(query_vector),
    )

    logger.info(
        "Embedding model test completed successfully."
    )