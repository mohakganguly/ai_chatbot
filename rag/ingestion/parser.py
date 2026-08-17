"""
parser.py

Enterprise Document Parser.

Responsibilities
----------------
1. Validate loader output.
2. Normalize metadata.
3. Preserve loader-generated metadata.
4. Ensure downstream compatibility.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentParser:
    """
    Normalizes documents produced by every loader.

    Since every loader already returns LangChain
    Documents, this parser only validates and
    standardizes metadata.
    """

    DEFAULT_METADATA = {

        "source": "unknown",

        "page": 1,

        "loader": "unknown",

        "ocr": False,

    }

    def parse(
        self,
        documents: List[Document],
    ) -> List[Document]:

        logger.info(
            "Parsing %d document(s).",
            len(documents),
        )

        start = time.perf_counter()

        parsed_documents = [

            self._parse_document(doc)

            for doc in documents

        ]

        logger.info(
            "Parsing completed in %.3f sec.",
            time.perf_counter() - start,
        )

        return parsed_documents

    # -----------------------------------------------------
    # Single Document
    # -----------------------------------------------------

    def _parse_document(
        self,
        document: Document,
    ) -> Document:

        metadata = self.DEFAULT_METADATA.copy()

        if isinstance(document.metadata, dict):

            metadata.update(document.metadata)

        # ---------------------------------------------
        # Normalize source
        # ---------------------------------------------

        source = metadata.get("source")

        if source:

            metadata["source"] = str(Path(source))

            metadata["filename"] = Path(source).name

            metadata["document_type"] = (

                Path(source)

                .suffix

                .replace(".", "")

                .lower()

            )

        else:

            metadata["filename"] = "unknown"

            metadata["document_type"] = "unknown"

        # ---------------------------------------------
        # Normalize page
        # ---------------------------------------------

        page = metadata.get("page")

        if page is None:

            metadata["page"] = 1

        else:

            try:

                metadata["page"] = int(page)

            except Exception:

                metadata["page"] = 1

        # ---------------------------------------------
        # Normalize OCR flag
        # ---------------------------------------------

        metadata["ocr"] = bool(

            metadata.get("ocr", False)

        )

        # ---------------------------------------------
        # Normalize loader name
        # ---------------------------------------------

        metadata["loader"] = str(

            metadata.get(

                "loader",

                "unknown",

            )

        )

        return Document(

            page_content=document.page_content,

            metadata=metadata,

        )


# ==========================================================
# Public API
# ==========================================================

_parser = DocumentParser()


def parse_documents(
    documents: List[Document],
) -> List[Document]:

    return _parser.parse(
        documents
    )


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    from rag.ingestion.loader import load_document

    docs = load_document(
        "documents/sample.pdf"
    )

    docs = parse_documents(
        docs
    )

    print("=" * 80)

    print(docs[0].metadata)

    print("=" * 80)

    print(docs[0].page_content[:500])