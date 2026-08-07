# """
# cleaner.py

# Cleans document text while preserving its meaning.
# """

# from __future__ import annotations

# import re
# import time
# import unicodedata
# from typing import List

# from langchain_core.documents import Document

# from utils.logger import get_logger

# logger = get_logger(__name__)


# class DocumentCleaner:
#     """
#     Cleans document text while preserving semantic meaning.
#     """

#     @classmethod
#     def clean(
#         cls,
#         documents: List[Document],
#     ) -> List[Document]:

#         logger.info("Cleaning %d document(s).", len(documents))

#         cleaned_documents = []

#         try:
#             for document in documents:
#                 cleaned_documents.append(
#                     cls._clean_single_document(document)
#                 )

#             logger.info(
#                 "Successfully cleaned %d document(s).",
#                 len(cleaned_documents),
#             )

#             return cleaned_documents

#         except Exception:
#             logger.exception("Failed while cleaning documents.")
#             raise

#     @classmethod
#     def _clean_single_document(cls, document: Document) -> Document:
#         text = document.page_content
#         is_ocr = document.metadata.get("ocr", False)
#         is_table = document.metadata.get("category") == "Table"

#         text = cls._normalize_unicode(text)
#         text = cls._remove_non_breaking_spaces(text)
#         text = cls._replace_tabs(text)

#         if is_ocr:
#             text = cls._dehyphenate(text)

#         if not is_table:
#             text = cls._remove_extra_spaces(text)
#             text = cls._remove_extra_newlines(text)

#         text = text.strip()
#         return Document(page_content=text, metadata=document.metadata)

#     @staticmethod
#     def _dehyphenate(text: str) -> str:
#         # join words broken across a line by OCR: "exam-\nple" -> "example"
#         return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

#     @staticmethod
#     def _normalize_unicode(text: str) -> str:

#         return unicodedata.normalize("NFKC", text)

#     @staticmethod
#     def _remove_non_breaking_spaces(text: str) -> str:

#         return text.replace("\xa0", " ")

#     @staticmethod
#     def _replace_tabs(text: str) -> str:

#         return text.replace("\t", " ")

#     @staticmethod
#     def _remove_extra_spaces(text: str) -> str:

#         return re.sub(r"[ ]{2,}", " ", text)

#     @staticmethod
#     def _remove_extra_newlines(text: str) -> str:

#         return re.sub(r"\n{3,}", "\n\n", text)


# def clean_documents(
#     documents: List[Document],
# ) -> List[Document]:

#     return DocumentCleaner.clean(documents)


# # ---------------------------------------------------------------------
# # Testing
# # ---------------------------------------------------------------------

# if __name__ == "__main__":

#     from rag.ingestion.loader import load_document
#     from rag.ingestion.parser import parse_documents

#     pdf_path = "documents/Must KNOW.pdf"

#     logger.info("Starting cleaner test.")

#     start = time.perf_counter()

#     docs = load_document(pdf_path)
#     docs = parse_documents(docs)

#     preprocessing_time = time.perf_counter() - start

#     logger.info(
#         "Loading + parsing completed in %.3f sec.",
#         preprocessing_time,
#     )

#     start = time.perf_counter()

#     cleaned_docs = clean_documents(docs)

#     cleaning_time = time.perf_counter() - start

#     logger.info(
#         "Cleaning completed in %.3f sec.",
#         cleaning_time,
#     )

#     if cleaned_docs:
#         logger.info(
#             "Sample metadata: %s",
#             cleaned_docs[0].metadata,
#         )

#         logger.info(
#             "Sample cleaned text (first 300 chars): %s",
#             cleaned_docs[0].page_content[:300],
#         )

#     logger.info("Cleaner test completed successfully.")



"""
cleaner.py

Enterprise Document Cleaner.

Responsibilities
----------------
1. Normalize unicode.
2. Preserve Markdown formatting.
3. Remove invisible characters.
4. Clean OCR artifacts.
5. Normalize whitespace.
6. Remove empty documents.
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
    Cleans extracted documents while preserving
    semantic structure for better chunking.
    """

    def clean(
        self,
        documents: List[Document],
    ) -> List[Document]:

        logger.info(
            "Cleaning %d document(s).",
            len(documents),
        )

        start = time.perf_counter()

        cleaned_documents = []

        for document in documents:

            cleaned = self._clean_document(
                document
            )

            if cleaned.page_content.strip():

                cleaned_documents.append(
                    cleaned
                )

        elapsed = time.perf_counter() - start

        logger.info(
            "Cleaning finished in %.3f sec.",
            elapsed,
        )

        logger.info(
            "Remaining documents : %d",
            len(cleaned_documents),
        )

        return cleaned_documents

    # -----------------------------------------------------
    # Single Document
    # -----------------------------------------------------

    def _clean_document(
        self,
        document: Document,
    ) -> Document:

        text = document.page_content

        text = self._normalize_unicode(text)

        text = self._remove_invisible_chars(text)

        text = self._normalize_line_endings(text)

        if document.metadata.get("ocr"):

            text = self._fix_ocr(text)

        text = self._normalize_whitespace(text)

        text = self._preserve_markdown(text)

        return Document(

            page_content=text,

            metadata=document.metadata,

        )

    # -----------------------------------------------------
    # Cleaning Steps
    # -----------------------------------------------------

    @staticmethod
    def _normalize_unicode(
        text: str,
    ) -> str:

        return unicodedata.normalize(
            "NFKC",
            text,
        )

    @staticmethod
    def _remove_invisible_chars(
        text: str,
    ) -> str:

        invisible = {

            "\u200b": "",

            "\u200c": "",

            "\u200d": "",

            "\ufeff": "",

        }

        for old, new in invisible.items():

            text = text.replace(
                old,
                new,
            )

        return text

    @staticmethod
    def _normalize_line_endings(
        text: str,
    ) -> str:

        return text.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

    # -----------------------------------------------------
    # OCR Cleanup
    # -----------------------------------------------------

    @staticmethod
    def _fix_ocr(
        text: str,
    ) -> str:

        # exam-
        # ple
        # ->
        # example

        text = re.sub(

            r"(\w)-\n(\w)",

            r"\1\2",

            text,

        )

        # remove isolated spaces

        text = re.sub(

            r"[ ]{2,}",

            " ",

            text,

        )

        return text

    # -----------------------------------------------------
    # Whitespace
    # -----------------------------------------------------

    @staticmethod
    def _normalize_whitespace(
        text: str,
    ) -> str:

        cleaned_lines = []

        for line in text.split("\n"):

            cleaned_lines.append(
                line.rstrip()
            )

        text = "\n".join(
            cleaned_lines
        )

        # collapse 4+ blank lines

        text = re.sub(

            r"\n{4,}",

            "\n\n\n",

            text,

        )

        return text.strip()

    # -----------------------------------------------------
    # Markdown Preservation
    # -----------------------------------------------------

    @staticmethod
    def _preserve_markdown(
        text: str,
    ) -> str:

        # preserve headings

        text = re.sub(

            r"\n([#]{1,6}\s)",

            r"\n\1",

            text,

        )

        # preserve bullets

        text = re.sub(

            r"\n([*-]\s)",

            r"\n\1",

            text,

        )

        return text


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

_cleaner = DocumentCleaner()


def clean_documents(
    documents: List[Document],
) -> List[Document]:

    return _cleaner.clean(
        documents
    )


# ---------------------------------------------------------
# Testing
# ---------------------------------------------------------

if __name__ == "__main__":

    from rag.ingestion.loader import load_document

    docs = load_document(
        "documents/sample.pdf"
    )

    docs = clean_documents(
        docs
    )

    print("=" * 80)

    print(len(docs))

    print("=" * 80)

    print(docs[0].metadata)

    print()

    print(docs[0].page_content[:1000])