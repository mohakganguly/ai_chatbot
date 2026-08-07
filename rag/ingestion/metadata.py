# """
# metadata.py

# Adds enterprise metadata to documents.
# """

# from __future__ import annotations

# import hashlib
# import time
# import uuid
# from datetime import datetime
# from typing import List

# from langchain_core.documents import Document

# from utils.logger import get_logger

# logger = get_logger(__name__)


# class MetadataBuilder:
#     """
#     Enriches documents with enterprise metadata required for indexing,
#     versioning, auditing, and traceability.
#     """

#     @classmethod
#     def enrich(
#         cls,
#         documents: List[Document],
#     ) -> List[Document]:

#         logger.info(
#             "Enriching metadata for %d document(s).",
#             len(documents),
#         )

#         document_id = str(uuid.uuid4())

#         enriched_documents = []

#         try:

#             for document in documents:

#                 enriched_documents.append(
#                     cls._enrich_single_document(
#                         document=document,
#                         document_id=document_id,
#                     )
#                 )

#             logger.info(
#                 "Successfully enriched %d document(s).",
#                 len(enriched_documents),
#             )

#             return enriched_documents

#         except Exception:
#             logger.exception("Failed while enriching document metadata.")
#             raise

#     @classmethod
#     def _enrich_single_document(
#         cls,
#         document: Document,
#         document_id: str,
#     ) -> Document:

#         metadata = document.metadata.copy()

#         text = document.page_content

#         now = datetime.now().isoformat()

#         metadata["document_id"] = document_id
#         metadata["chunk_id"] = None
#         metadata["version"] = 1
#         metadata["created_at"] = now
#         metadata["updated_at"] = now
#         metadata["char_count"] = len(text)
#         metadata["word_count"] = len(text.split())
#         metadata["content_hash"] = cls._hash(text)
#         metadata["indexed"] = False

#         return Document(
#             page_content=text,
#             metadata=metadata,
#         )

#     @staticmethod
#     def _hash(text: str) -> str:

#         return hashlib.sha256(
#             text.encode("utf-8")
#         ).hexdigest()


# def enrich_metadata(
#     documents: List[Document],
# ) -> List[Document]:

#     return MetadataBuilder.enrich(documents)


# # ---------------------------------------------------------------------
# # Testing
# # ---------------------------------------------------------------------

# if __name__ == "__main__":

#     from rag.ingestion.cleaner import clean_documents
#     from rag.ingestion.loader import load_document
#     from rag.ingestion.parser import parse_documents

#     pdf_path = "documents/Must KNOW.pdf"

#     logger.info("Starting metadata enrichment test.")

#     start = time.perf_counter()

#     docs = load_document(pdf_path)
#     docs = parse_documents(docs)
#     docs = clean_documents(docs)

#     preprocessing_time = time.perf_counter() - start

#     logger.info(
#         "Loading + parsing + cleaning completed in %.3f sec.",
#         preprocessing_time,
#     )

#     start = time.perf_counter()

#     enriched_docs = enrich_metadata(docs)

#     metadata_time = time.perf_counter() - start

#     logger.info(
#         "Metadata enrichment completed in %.3f sec.",
#         metadata_time,
#     )

#     if enriched_docs:

#         logger.info(
#             "Sample metadata: %s",
#             enriched_docs[0].metadata,
#         )

#         logger.info(
#             "Sample content (first 300 chars): %s",
#             enriched_docs[0].page_content[:300],
#         )

#     logger.info("Metadata enrichment test completed successfully.")


"""
metadata.py

Enterprise Metadata Builder.

Responsibilities
----------------
1. Preserve loader metadata.
2. Add indexing metadata.
3. Add retrieval metadata.
4. Add observability metadata.
5. Generate stable document identifiers.
"""

from __future__ import annotations

import hashlib
import time
import uuid

from datetime import datetime
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from utils.logger import get_logger

logger = get_logger(__name__)

class MetadataBuilder:

    def enrich(

        self,

        documents: List[Document],

    ) -> List[Document]:

        logger.info(

            "Enriching metadata for %d documents.",

            len(documents),

        )

        start = time.perf_counter()

        document_id = str(uuid.uuid4())

        enriched = [

            self._enrich_document(

                document,

                document_id,

            )

            for document in documents

        ]

        logger.info(

            "Metadata enrichment finished in %.3f sec.",

            time.perf_counter() - start,

        )

        return enriched

    def _enrich_document(

        self,

        document: Document,

        document_id: str,

    ) -> Document:

        metadata = document.metadata.copy()

        text = document.page_content

        now = datetime.utcnow().isoformat()

        metadata.setdefault(

            "document_id",

            document_id,

        )

        metadata.setdefault(

            "version",

            1,

        )

        metadata.setdefault(

            "loader",

            "unknown",

        )

        metadata.setdefault(

            "ocr",

            False,

        )

        metadata.setdefault(

            "page",

            1,

        )

        metadata.setdefault(

            "source",

            "unknown",

        )

        metadata["filename"] = Path(

            metadata["source"]

        ).name

        metadata["document_type"] = (

            Path(

                metadata["source"]

            ).suffix

            .replace(".", "")

            .lower()

        )

        metadata["indexed"] = False

        metadata["char_count"] = len(text)

        metadata["word_count"] = len(

            text.split()

        )

        metadata["line_count"] = len(

            text.splitlines()

        )
        metadata["content_hash"] = (

            self._hash(

                text

            )

        )

        metadata["created_at"] = now

        metadata["updated_at"] = now

        metadata.setdefault(

            "chunk_id",

            None,

        )
        return Document(

            page_content=text,

            metadata=metadata,

        )

    @staticmethod

    def _hash(

        text: str,

    ) -> str:

        return hashlib.sha256(

            text.encode(

                "utf-8"

            )

        ).hexdigest()

_builder = MetadataBuilder()


def enrich_metadata(

    documents: List[Document],

) -> List[Document]:

    return _builder.enrich(

        documents

    )