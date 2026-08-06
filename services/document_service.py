"""
document_service.py

Enterprise Document Service.

Responsible for the complete lifecycle of chat documents.

Responsibilities

Upload
↓

Save locally

↓

Create database record

↓

Index into vector database

↓

Retrieve documents

↓

Delete documents

↓

Re-index documents
"""

from __future__ import annotations

from pathlib import Path

from db.repository import (
    save_document,
    get_documents,
    delete_document,
)

from services.ingestion_service import (
    IngestionService,
)

from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentService:

    def __init__(self):

        self.ingestion = IngestionService()

    # ---------------------------------------------------------
    # Upload
    # ---------------------------------------------------------

    def upload_documents(
        self,
        thread_id: str,
        uploaded_files,
    ):

        logger.info(
            "Uploading %d document(s).",
            len(uploaded_files),
        )

        saved_paths = self.ingestion.save_uploaded_files(
            thread_id=thread_id,
            uploaded_files=uploaded_files,
        )

        self.ingestion.ingest_documents(
            thread_id=thread_id,
            file_paths=saved_paths,
        )

        return get_documents(thread_id)

    # ---------------------------------------------------------
    # Get Documents
    # ---------------------------------------------------------

    def list_documents(
        self,
        thread_id: str,
    ):

        return get_documents(thread_id)

    # ---------------------------------------------------------
    # Delete
    # ---------------------------------------------------------

    def remove_document(
        self,
        document,
    ):

        path = Path(document.filepath)

        if path.exists():

            path.unlink()

        delete_document(document.id)

        logger.info(
            "Deleted document %s",
            document.filename,
        )

    # ---------------------------------------------------------
    # Delete All
    # ---------------------------------------------------------

    def remove_all_documents(
        self,
        thread_id: str,
    ):

        documents = get_documents(thread_id)

        for document in documents:

            self.remove_document(document)