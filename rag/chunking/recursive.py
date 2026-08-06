"""
recursive.py

Splits documents into overlapping chunks using
RecursiveCharacterTextSplitter.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE
from utils.logger import get_logger

from rag.chunking.base import BaseChunker

logger = get_logger(__name__)


class RecursiveChunker(BaseChunker):
    """
    Splits documents into overlapping chunks using
    RecursiveCharacterTextSplitter.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):

        logger.info(
            "Initializing RecursiveChunker (chunk_size=%d, chunk_overlap=%d).",
            chunk_size,
            chunk_overlap,
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def chunk(
            self,
            documents: list[Document]
    ):

        chunks = self.splitter.split_documents(
            documents
        )

        return self._add_chunk_metadata(
            chunks
        )

    @staticmethod
    def _add_chunk_metadata(
        chunks: List[Document],
    ) -> List[Document]:

        page_chunk_counter = defaultdict(int)
        page_total_counter = defaultdict(int)

        # Count total chunks per page
        for chunk in chunks:
            page = chunk.metadata.get("page", -1)
            page_total_counter[page] += 1

        # Add metadata
        for chunk in chunks:

            metadata = chunk.metadata.copy()

            page = metadata.get("page", -1)

            metadata["chunk_id"] = str(uuid.uuid4())
            metadata["chunk_index"] = page_chunk_counter[page]
            metadata["total_chunks_in_page"] = page_total_counter[page]

            page_chunk_counter[page] += 1

            chunk.metadata = metadata

        return chunks


def chunk_documents(
    documents: List[Document],
):

    chunker = RecursiveChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return chunker.chunk(documents)


# ---------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------

if __name__ == "__main__":

    from rag.ingestion.cleaner import clean_documents
    from rag.ingestion.loader import load_document
    from rag.ingestion.metadata import enrich_metadata
    from rag.ingestion.parser import parse_documents

    pdf_path = "documents/Must KNOW.pdf"

    logger.info("Starting recursive chunker test.")

    start = time.perf_counter()

    docs = load_document(pdf_path)
    docs = parse_documents(docs)
    docs = clean_documents(docs)
    docs = enrich_metadata(docs)

    preprocessing_time = time.perf_counter() - start

    logger.info(
        "Preprocessing completed in %.3f sec.",
        preprocessing_time,
    )

    start = time.perf_counter()

    chunks = chunk_documents(docs)

    chunking_time = time.perf_counter() - start

    logger.info(
        "Chunking completed in %.3f sec.",
        chunking_time,
    )

    logger.info(
        "Total chunks created: %d",
        len(chunks),
    )

    if chunks:

        logger.info(
            "Sample chunk metadata: %s",
            chunks[0].metadata,
        )

        logger.info(
            "Sample chunk content (first 300 chars): %s",
            chunks[0].page_content[:300],
        )

    logger.info("Recursive chunker test completed successfully.")