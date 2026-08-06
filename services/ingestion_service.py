"""
ingestion_service.py

Business logic for indexing documents into the knowledge base.

This service is used by the Streamlit frontend (and later FastAPI)
to process uploaded documents.

Pipeline

Upload
    ↓
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
Embedding
    ↓
Qdrant
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from utils.logger import get_logger

from rag.ingestion.loader import load_document
from rag.ingestion.parser import parse_documents
from rag.ingestion.cleaner import clean_documents
from rag.ingestion.metadata import enrich_metadata

from rag.chunking.recursive import chunk_documents

from rag.embedding.embedding_model import get_embedding_model

from rag.vectorstore.vector_store import QdrantVectorStore

logger = get_logger(__name__)

# ---------------------------------------------------------
# Directory where uploaded files are stored
# ---------------------------------------------------------

DOCUMENTS_DIR = Path("documents")

DOCUMENTS_DIR.mkdir(
    exist_ok=True
)


class IngestionService:

    def __init__(self):

        logger.info(
            "Initializing Ingestion Service"
        )

        self.embedder = get_embedding_model()

        self.vector_store = QdrantVectorStore()

        self.vector_store.create_collection()

    # -----------------------------------------------------
    # Save uploaded files
    # -----------------------------------------------------

    def save_uploaded_files(
        self,
        uploaded_files,
    ) -> List[Path]:

        saved_paths = []

        for uploaded_file in uploaded_files:

            destination = (
                DOCUMENTS_DIR /
                uploaded_file.name
            )

            with open(destination, "wb") as f:
                shutil.copyfileobj(
                    uploaded_file,
                    f
                )

            logger.info(
                "Saved %s",
                destination
            )

            saved_paths.append(destination)

        return saved_paths

    # -----------------------------------------------------
    # Index Documents
    # -----------------------------------------------------

    def ingest_documents(
        self,
        file_paths: List[Path],
    ):

        logger.info(
            "Starting ingestion of %d document(s)",
            len(file_paths)
        )

        total_chunks = 0

        for path in file_paths:

            logger.info(
                "Processing %s",
                path.name
            )

            docs = load_document(str(path))

            docs = parse_documents(docs)

            docs = clean_documents(docs)

            docs = enrich_metadata(docs)

            chunks = chunk_documents(docs)

            embedded_chunks = (
                self.embedder.embed_documents(
                    chunks
                )
            )

            self.vector_store.upsert_documents(
                embedded_chunks
            )

            total_chunks += len(chunks)

            logger.info(
                "%s indexed successfully",
                path.name
            )

        logger.info(
            "Finished indexing %d chunks.",
            total_chunks
        )

        return {
            "documents": len(file_paths),
            "chunks": total_chunks,
        }