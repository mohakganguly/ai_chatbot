"""
cleaner.py

Cleans document text while preserving its meaning.
"""

from __future__ import annotations

import re
import time
import unicodedata
from typing import List

from langchain_core.documents import Document

from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentCleaner:
    """
    Cleans document text while preserving semantic meaning.
    """

    @classmethod
    def clean(
        cls,
        documents: List[Document],
    ) -> List[Document]:

        logger.info("Cleaning %d document(s).", len(documents))

        cleaned_documents = []

        try:
            for document in documents:
                cleaned_documents.append(
                    cls._clean_single_document(document)
                )

            logger.info(
                "Successfully cleaned %d document(s).",
                len(cleaned_documents),
            )

            return cleaned_documents

        except Exception:
            logger.exception("Failed while cleaning documents.")
            raise

    @classmethod
    def _clean_single_document(
        cls,
        document: Document,
    ) -> Document:

        text = document.page_content

        text = cls._normalize_unicode(text)
        text = cls._remove_non_breaking_spaces(text)
        text = cls._replace_tabs(text)
        text = cls._remove_extra_spaces(text)
        text = cls._remove_extra_newlines(text)
        text = text.strip()

        return Document(
            page_content=text,
            metadata=document.metadata,
        )

    @staticmethod
    def _normalize_unicode(text: str) -> str:

        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _remove_non_breaking_spaces(text: str) -> str:

        return text.replace("\xa0", " ")

    @staticmethod
    def _replace_tabs(text: str) -> str:

        return text.replace("\t", " ")

    @staticmethod
    def _remove_extra_spaces(text: str) -> str:

        return re.sub(r"[ ]{2,}", " ", text)

    @staticmethod
    def _remove_extra_newlines(text: str) -> str:

        return re.sub(r"\n{3,}", "\n\n", text)


def clean_documents(
    documents: List[Document],
) -> List[Document]:

    return DocumentCleaner.clean(documents)


# ---------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------

if __name__ == "__main__":

    from rag.ingestion.loader import load_document
    from rag.ingestion.parser import parse_documents

    pdf_path = "documents/Must KNOW.pdf"

    logger.info("Starting cleaner test.")

    start = time.perf_counter()

    docs = load_document(pdf_path)
    docs = parse_documents(docs)

    preprocessing_time = time.perf_counter() - start

    logger.info(
        "Loading + parsing completed in %.3f sec.",
        preprocessing_time,
    )

    start = time.perf_counter()

    cleaned_docs = clean_documents(docs)

    cleaning_time = time.perf_counter() - start

    logger.info(
        "Cleaning completed in %.3f sec.",
        cleaning_time,
    )

    if cleaned_docs:
        logger.info(
            "Sample metadata: %s",
            cleaned_docs[0].metadata,
        )

        logger.info(
            "Sample cleaned text (first 300 chars): %s",
            cleaned_docs[0].page_content[:300],
        )

    logger.info("Cleaner test completed successfully.")