"""
pipeline.py

Enterprise document indexing pipeline.

Flow:
Loader
    ↓
Parser
    ↓
Cleaner
    ↓
Metadata
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Store
"""

from __future__ import annotations

from pathlib import Path

from utils.logger import get_logger

from rag.ingestion.loader import load_document
from rag.ingestion.parser import parse_documents
from rag.ingestion.cleaner import clean_documents
from rag.ingestion.metadata import enrich_metadata

from rag.chunking.factory import ChunkingFactory

from rag.embedding.embedding_model import (
    get_embedding_model,
)

from rag.vectorstore.vector_store import (
    QdrantVectorStore,
)

logger = get_logger(__name__)


class IndexingPipeline:

    def __init__(self):

        logger.info("Initializing Indexing Pipeline")

        self.embedder = get_embedding_model()

        self.vector_store = QdrantVectorStore()

        self.vector_store.create_collection()

        self.chunker = ChunkingFactory.create()

    def ingest(
        self,
        file_path: str
    ):

        logger.info("=" * 80)
        logger.info("Starting ingestion for %s", file_path)

        if not Path(file_path).exists():

            logger.error("File not found : %s", file_path)

            raise FileNotFoundError(file_path)

        # --------------------------------------------------

        logger.info("Step 1 : Loading document")

        documents = load_document(file_path)

        # --------------------------------------------------

        logger.info("Step 2 : Parsing document")

        documents = parse_documents(documents)

        # --------------------------------------------------

        logger.info("Step 3 : Cleaning document")

        documents = clean_documents(documents)

        # --------------------------------------------------

        logger.info("Step 4 : Adding metadata")

        documents = enrich_metadata(documents)

        # --------------------------------------------------

        logger.info("Step 5 : Chunking")

        chunks = self.chunker.chunk(
            documents
        )

        logger.info(
            "Created %d chunks",
            len(chunks)
        )

        # --------------------------------------------------

        logger.info("Step 6 : Generating embeddings")

        embedded_chunks = self.embedder.embed_documents(
            chunks
        )

        logger.info(
            "Generated %d embeddings",
            len(embedded_chunks)
        )

        # --------------------------------------------------

        logger.info("Step 7 : Uploading to Vector Store")

        self.vector_store.upsert_documents(
            embedded_chunks
        )

        logger.info(
            "Successfully indexed '%s'",
            file_path
        )

        logger.info("=" * 80)

        return embedded_chunks

if __name__ == "__main__":

    pipeline = IndexingPipeline()

    pipeline.ingest(
        "documents/Must KNOW.pdf"
    )