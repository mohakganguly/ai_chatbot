"""
vectorstore.py

Central wrapper around Qdrant.
"""

from __future__ import annotations

import time

from typing import List

from langchain_core.documents import Document

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

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

    def search(
        self,
        query_vector: List[float],
        thread_id: str|None = None,
        limit: int = 5,
    ) -> List[Document]:

        logger.info(
            "Searching top-%d document(s).",
            limit,
        )

        start = time.perf_counter()

        try:

            # filters = []

            # if thread_id is not None:

            #     filters.append(
            #         FieldCondition(
            #             key="thread_id",
            #             match=MatchValue(value=thread_id),
            #         )
            #     )

            query_filter = (
                Filter(
                    must=[
                        FieldCondition(
                            key="thread_id",
                            match=MatchValue(value=thread_id),
                        )
                    ]
                )
                if thread_id
                else None
            )
            logger.info(
                "Searching collection '%s' for thread '%s'",
                COLLECTION_NAME,
                thread_id,
            )
            
            results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
            ).points

            documents = []

            for result in results:

                payload = dict(result.payload)

                text = payload.pop("text")

                payload["retrieval_score"] = result.score
                documents.append(
                    Document(
                        page_content=text,
                        metadata=payload,
                    )
                )

            elapsed = time.perf_counter() - start

            logger.info(
                "Retrieved %d document(s) in %.3f sec.",
                len(documents),
                elapsed,
            )

            return documents

        except Exception:

            logger.exception(
                "Vector search failed."
            )

            raise

    def create_collection(self):

        logger.info(
            "Checking collection '%s'.",
            COLLECTION_NAME,
        )

        start = time.perf_counter()

        try:

            collections = self.client.get_collections().collections

            names = [c.name for c in collections]

            if COLLECTION_NAME in names:

                logger.info(
                    "Collection '%s' already exists.",
                    COLLECTION_NAME,
                )

                return

            distance = (
                Distance.COSINE
                if DISTANCE_METRIC.upper() == "COSINE"
                else Distance.EUCLID
            )

            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=distance,
                ),
            )

            elapsed = time.perf_counter() - start

            logger.info(
                "Collection '%s' created successfully in %.3f sec.",
                COLLECTION_NAME,
                elapsed,
            )

        except Exception:

            logger.exception(
                "Failed to create collection '%s'.",
                COLLECTION_NAME,
            )

            raise

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

    def upsert_documents(
        self,
        chunks: list[EmbeddedChunk]
    ):

        logger.info(
            "Upserting %d embedded chunk(s).",
            len(chunks),
        )

        start = time.perf_counter()

        try:

            points = []

            for chunk in chunks:

                document = chunk.document

                vector = chunk.vector

                metadata = document.metadata.copy()

                payload = {
                    "text": document.page_content,
                    **metadata,
                }

                points.append(
                    PointStruct(
                        id=metadata["chunk_id"],
                        vector=vector,
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=COLLECTION_NAME,
                wait=True,
                points=points,
            )

            elapsed = time.perf_counter() - start

            logger.info(
                "Successfully inserted %d vector(s) in %.3f sec.",
                len(points),
                elapsed,
            )

        except Exception:

            logger.exception(
                "Failed to upsert vectors into collection '%s'.",
                COLLECTION_NAME,
            )

            raise

if __name__ == "__main__":

    from rag.ingestion.loader import load_document
    from rag.ingestion.parser import parse_documents
    from rag.ingestion.cleaner import clean_documents
    from rag.ingestion.metadata import enrich_metadata
    from rag.chunking.chunker import chunk_documents
    from rag.embedding.embedding_model import get_embedding_model

    pdf_path = "documents/Must KNOW.pdf"

    logger.info("Starting Qdrant vector store integration test.")

    # ------------------------------------------------------------
    # Document Processing Pipeline
    # ------------------------------------------------------------

    start = time.perf_counter()

    docs = load_document(pdf_path)

    docs = parse_documents(docs)

    docs = clean_documents(docs)

    docs = enrich_metadata(docs)

    chunks = chunk_documents(docs)

    preprocessing_time = time.perf_counter() - start

    logger.info(
            "Document preprocessing completed in %.3f sec.",
            preprocessing_time,
    )

    logger.info(
        "Generated %d chunk(s).",
        len(chunks),
    )

    # ------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------

    embedder = get_embedding_model()

    start = time.perf_counter()

    embedded_chunks = embedder.embed_documents(chunks)

    embedding_time = time.perf_counter() - start

    logger.info(
        "Generated %d embedding(s) in %.3f sec.",
        len(embedded_chunks),
        embedding_time,
    )

    # ------------------------------------------------------------
    # Vector Store
    # ------------------------------------------------------------

    store = QdrantVectorStore()

    store.create_collection()

    info = store.collection_info()

    logger.info(
        "Collection information:\n%s",
        info,
    )

    start = time.perf_counter()

    store.upsert_documents(embedded_chunks)

    upsert_time = time.perf_counter() - start

    logger.info(
        "Vector upsert completed in %.3f sec.",
        upsert_time,
    )

    # ------------------------------------------------------------
    # Similarity Search
    # ------------------------------------------------------------

    query = "What is a Zombie Process?"

    logger.info(
        "Running similarity search for query: %s",
        query,
    )

    query_vector = embedder.embed_query(query)

    start = time.perf_counter()

    results = store.search(query_vector)

    search_time = time.perf_counter() - start

    logger.info(
        "Similarity search completed in %.3f sec.",
        search_time,
    )

    logger.info(
        "Retrieved %d document(s).",
        len(results),
    )

    for index, document in enumerate(results, start=1):
        logger.info(
            "Result %d Metadata: %s",
            index,
            document.metadata,
        )

        logger.info(
            "Result %d Content (first 300 chars): %s",
            index,
            document.page_content[:300],
        )

    logger.info(
            "Qdrant vector store integration test completed successfully."
        )