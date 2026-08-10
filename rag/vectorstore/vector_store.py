"""
vector_store.py

Hybrid Qdrant vector store.

Retrieval architecture:

Dense BGE embedding
        +
Qdrant BM25 sparse retrieval
        ↓
       RRF
        ↓
Hybrid candidates

Responsibilities:
1. Connect to Qdrant
2. Create hybrid collection
3. Store dense + BM25 vectors
4. Perform hybrid retrieval
5. Apply thread-level filtering
6. Return LangChain Documents
"""

from __future__ import annotations

import time
from typing import List

from langchain_core.documents import Document

from qdrant_client import QdrantClient, models

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    VECTOR_SIZE,
    DISTANCE_METRIC,
)

from rag.models.embedded_chunk import EmbeddedChunk
from utils.logger import get_logger


logger = get_logger(__name__)


# ==========================================================
# Vector names
# ==========================================================

DENSE_VECTOR_NAME = "dense"
BM25_VECTOR_NAME = "bm25"


# ==========================================================
# Retrieval configuration
# ==========================================================

DENSE_TOP_K = 15
SPARSE_TOP_K = 15
HYBRID_TOP_K = 10


# ==========================================================
# Qdrant Vector Store
# ==========================================================

class QdrantVectorStore:

    def __init__(self):

        logger.info(
            "Connecting to Qdrant (%s:%s).",
            QDRANT_HOST,
            QDRANT_PORT,
        )

        try:

            self.client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
            )

            logger.info(
                "Connected to Qdrant successfully."
            )

        except Exception:

            logger.exception(
                "Failed to connect to Qdrant."
            )

            raise


    # ======================================================
    # Collection
    # ======================================================

    def create_collection(self):

        logger.info(
            "Checking hybrid collection '%s'.",
            COLLECTION_NAME,
        )

        try:

            collections = (
                self.client
                .get_collections()
                .collections
            )

            names = [
                collection.name
                for collection in collections
            ]

            if COLLECTION_NAME in names:

                logger.info(
                    "Collection '%s' already exists.",
                    COLLECTION_NAME,
                )

                return

            distance = (
                models.Distance.COSINE
                if DISTANCE_METRIC.upper() == "COSINE"
                else models.Distance.EUCLID
            )

            logger.info(
                "Creating hybrid collection."
            )

            self.client.create_collection(

                collection_name=COLLECTION_NAME,

                vectors_config={

                    DENSE_VECTOR_NAME:
                        models.VectorParams(

                            size=VECTOR_SIZE,

                            distance=distance,
                        ),
                },

                sparse_vectors_config={

                    BM25_VECTOR_NAME:
                        models.SparseVectorParams(

                            modifier=models.Modifier.IDF,
                        ),
                },
            )

            logger.info(
                "Hybrid collection '%s' created successfully.",
                COLLECTION_NAME,
            )

        except Exception:

            logger.exception(
                "Failed to create collection '%s'.",
                COLLECTION_NAME,
            )

            raise


    # ======================================================
    # Collection migration
    # ======================================================

    def recreate_collection(self):

        """
        Deletes the current collection and recreates it
        with dense + BM25 vectors.

        WARNING:
        This deletes all existing points.

        Existing documents must be re-indexed afterwards.
        """

        logger.warning(
            "Recreating collection '%s'.",
            COLLECTION_NAME,
        )

        if self.client.collection_exists(
            COLLECTION_NAME
        ):

            self.client.delete_collection(
                COLLECTION_NAME
            )

            logger.info(
                "Old collection deleted."
            )

        self.create_collection()

        logger.info(
            "Hybrid collection recreated successfully."
        )


    # ======================================================
    # Collection info
    # ======================================================

    def collection_info(self):

        logger.info(
            "Fetching collection information."
        )

        try:

            info = self.client.get_collection(
                COLLECTION_NAME
            )

            logger.info(
                "Collection information retrieved successfully."
            )

            return info

        except Exception:

            logger.exception(
                "Failed to fetch collection information."
            )

            raise


    # ======================================================
    # Hybrid Search
    # ======================================================

    def search(
        self,
        query: str,
        query_vector: List[float],
        thread_id: str | None = None,
        limit: int = HYBRID_TOP_K,
    ) -> List[Document]:

        logger.info(
            "Hybrid search started."
        )

        logger.info(
            "Dense candidates : %d",
            DENSE_TOP_K,
        )

        logger.info(
            "BM25 candidates  : %d",
            SPARSE_TOP_K,
        )

        logger.info(
            "RRF results      : %d",
            limit,
        )

        start = time.perf_counter()

        try:

            # --------------------------------------------------
            # Thread filter
            # --------------------------------------------------

            query_filter = (

                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="thread_id",
                            match=models.MatchValue(
                                value=thread_id
                            ),
                        )
                    ]
                )

                if thread_id

                else None
            )

            # --------------------------------------------------
            # Dense retrieval
            # --------------------------------------------------

            dense_prefetch = models.Prefetch(

                query=query_vector,

                using=DENSE_VECTOR_NAME,

                limit=DENSE_TOP_K,
            )

            # --------------------------------------------------
            # BM25 retrieval
            #
            # Qdrant generates the sparse representation
            # server-side.
            # --------------------------------------------------

            sparse_prefetch = models.Prefetch(

                query=models.Document(

                    text=query,

                    model="Qdrant/bm25",
                ),

                using=BM25_VECTOR_NAME,

                limit=SPARSE_TOP_K,
            )

            # --------------------------------------------------
            # RRF Fusion
            # --------------------------------------------------

            results = self.client.query_points(

                collection_name=COLLECTION_NAME,

                prefetch=[
                    dense_prefetch,
                    sparse_prefetch,
                ],

                query=models.FusionQuery(

                    fusion=models.Fusion.RRF
                ),

                query_filter=query_filter,

                limit=limit,

                with_payload=True,
            ).points

            # --------------------------------------------------
            # Convert to LangChain Documents
            # --------------------------------------------------

            documents = []

            for result in results:

                payload = dict(
                    result.payload or {}
                )

                text = payload.pop(
                    "text",
                    "",
                )

                payload["retrieval_score"] = (
                    float(result.score)
                    if result.score is not None
                    else 0.0
                )

                payload["retrieval_method"] = "hybrid_rrf"

                documents.append(

                    Document(

                        page_content=text,

                        metadata=payload,
                    )
                )

            elapsed = (
                time.perf_counter()
                - start
            )

            logger.info(
                "Hybrid retrieval returned %d document(s) in %.3f sec.",
                len(documents),
                elapsed,
            )

            return documents

        except Exception:

            logger.exception(
                "Hybrid Qdrant search failed."
            )

            raise


    # ======================================================
    # Upsert
    # ======================================================

    def upsert_documents(
        self,
        chunks: list[EmbeddedChunk],
    ):

        logger.info(
            "Upserting %d embedded chunk(s).",
            len(chunks),
        )

        if not chunks:

            logger.warning(
                "No chunks supplied for upsert."
            )

            return

        start = time.perf_counter()

        try:

            points = []

            for chunk in chunks:

                document = chunk.document

                dense_vector = chunk.vector

                metadata = (
                    document.metadata.copy()
                )

                text = document.page_content

                payload = {

                    "text": text,

                    **metadata,
                }

                # --------------------------------------------------
                # Dense + BM25
                #
                # BM25 is generated by Qdrant from the text.
                # --------------------------------------------------

                point = models.PointStruct(

                    id=metadata["chunk_id"],

                    vector={

                        DENSE_VECTOR_NAME:
                            dense_vector,

                        BM25_VECTOR_NAME:
                            models.Document(

                                text=text,

                                model="Qdrant/bm25",
                            ),
                    },

                    payload=payload,
                )

                points.append(point)

            # --------------------------------------------------
            # Upsert
            # --------------------------------------------------

            self.client.upsert(

                collection_name=COLLECTION_NAME,

                wait=True,

                points=points,
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            logger.info(
                "Successfully inserted %d hybrid vector point(s) in %.3f sec.",
                len(points),
                elapsed,
            )

        except Exception:

            logger.exception(
                "Failed to upsert hybrid vectors."
            )

            raise


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    from rag.ingestion.loader import load_document
    from rag.ingestion.parser import parse_documents
    from rag.ingestion.cleaner import clean_documents
    from rag.ingestion.metadata import enrich_metadata
    from rag.chunking.chunker import chunk_documents
    from rag.embedding.embedding_model import (
        get_embedding_model,
    )

    pdf_path = "documents/Must KNOW.pdf"

    logger.info(
        "Starting hybrid Qdrant integration test."
    )

    # ------------------------------------------------------
    # Document processing
    # ------------------------------------------------------

    docs = load_document(pdf_path)

    docs = parse_documents(docs)

    docs = clean_documents(docs)

    docs = enrich_metadata(docs)

    chunks = chunk_documents(docs)

    logger.info(
        "Generated %d chunk(s).",
        len(chunks),
    )

    # ------------------------------------------------------
    # Dense embeddings
    # ------------------------------------------------------

    embedder = get_embedding_model()

    embedded_chunks = (
        embedder.embed_documents(
            chunks
        )
    )

    # ------------------------------------------------------
    # Vector store
    # ------------------------------------------------------

    store = QdrantVectorStore()

    store.create_collection()

    info = store.collection_info()

    logger.info(
        "Collection information:\n%s",
        info,
    )

    # ------------------------------------------------------
    # Upsert
    # ------------------------------------------------------

    store.upsert_documents(
        embedded_chunks
    )

    # ------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------

    query = "What is a Zombie Process?"

    query_vector = (
        embedder.embed_query(query)
    )

    results = store.search(

        query=query,

        query_vector=query_vector,

        limit=HYBRID_TOP_K,
    )

    logger.info(
        "Retrieved %d hybrid result(s).",
        len(results),
    )

    for index, document in enumerate(
        results,
        start=1,
    ):

        logger.info(
            "Result %d metadata: %s",
            index,
            document.metadata,
        )

        logger.info(
            "Result %d content: %s",
            index,
            document.page_content[:300],
        )

    logger.info(
        "Hybrid Qdrant integration test completed."
    )