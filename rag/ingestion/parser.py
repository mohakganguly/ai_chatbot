"""
parser.py

Normalizes the output produced by different document loaders
into a common format used throughout the RAG pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentParser:
    """
    Converts loader output into a standardized format.
    """

    REQUIRED_METADATA = {
        "source": None,
        "page": None,
        "document_type": None,
    }

    @classmethod
    def parse(
        cls,
        documents: List[Document],
    ) -> List[Document]:

        logger.info("Parsing %d document(s).", len(documents))

        parsed_documents = []

        try:
            for document in documents:
                parsed_documents.append(
                    cls._parse_single_document(document)
                )

            logger.info(
                "Successfully parsed %d document(s).",
                len(parsed_documents),
            )

            return parsed_documents

        except Exception:
            logger.exception("Failed while parsing documents.")
            raise

    @classmethod
    def _parse_single_document(
        cls,
        document: Document,
    ) -> Document:

        metadata = cls._normalize_metadata(
            document.metadata
        )

        return Document(
            page_content=document.page_content,
            metadata=metadata,
        )

    @classmethod
    def _normalize_metadata(
        cls,
        metadata: dict,
    ) -> dict:

        normalized = cls.REQUIRED_METADATA.copy()

        normalized.update(metadata)

        source = normalized.get("source")

        if source:

            normalized["source"] = Path(source).name

            normalized["document_type"] = (
                Path(source)
                .suffix
                .replace(".", "")
                .lower()
            )

        if normalized.get("page") is None:
            normalized["page"] = None

        return normalized


def parse_documents(
    documents: List[Document],
) -> List[Document]:

    return DocumentParser.parse(documents)


# ---------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------

if __name__ == "__main__":

    import time

    from rag.ingestion.loader import load_document

    pdf_path = "documents/Must KNOW.pdf"

    logger.info("Starting parser test.")

    start = time.perf_counter()

    docs = load_document(pdf_path)

    loader_time = time.perf_counter() - start

    logger.info(
        "Loader completed in %.3f sec.",
        loader_time,
    )

    start = time.perf_counter()

    parsed_docs = parse_documents(docs)

    parser_time = time.perf_counter() - start

    logger.info(
        "Parser completed in %.3f sec.",
        parser_time,
    )

    if parsed_docs:
        logger.info(
            "Sample metadata: %s",
            parsed_docs[0].metadata,
        )

    logger.info("Parser test completed successfully.")