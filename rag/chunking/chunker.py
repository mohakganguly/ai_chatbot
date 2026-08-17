# """
# recursive.py

# Splits documents into overlapping chunks using
# RecursiveCharacterTextSplitter.
# """

# from __future__ import annotations

# import time
# import uuid
# from collections import defaultdict
# from typing import List

# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# from config import CHUNK_OVERLAP, CHUNK_SIZE
# from utils.logger import get_logger

# from rag.chunking.base import BaseChunker

# logger = get_logger(__name__)


# class RecursiveChunker(BaseChunker):
#     """
#     Splits documents into overlapping chunks using
#     RecursiveCharacterTextSplitter.
#     """

#     def __init__(
#         self,
#         chunk_size: int = 1000,
#         chunk_overlap: int = 200,
#     ):

#         logger.info(
#             "Initializing RecursiveChunker (chunk_size=%d, chunk_overlap=%d).",
#             chunk_size,
#             chunk_overlap,
#         )

#         self.splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             separators=[
#                 "\n\n",
#                 "\n",
#                 ". ",
#                 " ",
#                 "",
#             ],
#         )

#     def chunk(
#             self,
#             documents: list[Document]
#     ):

#         chunks = self.splitter.split_documents(
#             documents
#         )

#         return self._add_chunk_metadata(
#             chunks
#         )

#     @staticmethod
#     def _add_chunk_metadata(
#         chunks: List[Document],
#     ) -> List[Document]:

#         page_chunk_counter = defaultdict(int)
#         page_total_counter = defaultdict(int)

#         # Count total chunks per page
#         for chunk in chunks:
#             page = chunk.metadata.get("page", -1)
#             page_total_counter[page] += 1

#         # Add metadata
#         for chunk in chunks:

#             metadata = chunk.metadata.copy()

#             page = metadata.get("page", -1)

#             metadata["chunk_id"] = str(uuid.uuid4())
#             metadata["chunk_index"] = page_chunk_counter[page]
#             metadata["total_chunks_in_page"] = page_total_counter[page]

#             page_chunk_counter[page] += 1

#             chunk.metadata = metadata

#         return chunks


# def chunk_documents(
#     documents: List[Document],
# ):

#     chunker = RecursiveChunker(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#     )

#     return chunker.chunk(documents)


# # ---------------------------------------------------------------------
# # Testing
# # ---------------------------------------------------------------------

# if __name__ == "__main__":

#     from rag.ingestion.cleaner import clean_documents
#     from rag.ingestion.loader import load_document
#     from rag.ingestion.metadata import enrich_metadata
#     from rag.ingestion.parser import parse_documents

#     pdf_path = "documents/Must KNOW.pdf"

#     logger.info("Starting recursive chunker test.")

#     start = time.perf_counter()

#     docs = load_document(pdf_path)
#     docs = parse_documents(docs)
#     docs = clean_documents(docs)
#     docs = enrich_metadata(docs)

#     preprocessing_time = time.perf_counter() - start

#     logger.info(
#         "Preprocessing completed in %.3f sec.",
#         preprocessing_time,
#     )

#     start = time.perf_counter()

#     chunks = chunk_documents(docs)

#     chunking_time = time.perf_counter() - start

#     logger.info(
#         "Chunking completed in %.3f sec.",
#         chunking_time,
#     )

#     logger.info(
#         "Total chunks created: %d",
#         len(chunks),
#     )

#     if chunks:

#         logger.info(
#             "Sample chunk metadata: %s",
#             chunks[0].metadata,
#         )

#         logger.info(
#             "Sample chunk content (first 300 chars): %s",
#             chunks[0].page_content[:300],
#         )

#     logger.info("Recursive chunker test completed successfully.")

"""
recursive.py

Enterprise Hybrid Chunker.

Responsibilities
----------------

1. Fast recursive chunking for normal text.
2. Preserve Markdown headings and sections.
3. Preserve Markdown tables.
4. Preserve code blocks.
5. Handle large tables without destroying row structure.
6. Handle large code blocks without arbitrary character splitting.
7. Preserve OCR documents.
8. Preserve existing document metadata.
9. Add chunk-level metadata required by the RAG pipeline.

No LLM calls are made during chunking.
"""

from __future__ import annotations

import re
import time
import uuid
from collections import defaultdict
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE
from rag.chunking.base import BaseChunker

from utils.logger import get_logger

logger = get_logger(__name__)


# ==========================================================
# Regex Patterns
# ==========================================================

# Markdown fenced code block

CODE_BLOCK_PATTERN = re.compile(
    r"```[\s\S]*?```",
    re.MULTILINE,
)


# Markdown table

TABLE_LINE_PATTERN = re.compile(
    r"^\s*\|.*\|\s*$"
)

TABLE_SEPARATOR_PATTERN = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


# Markdown headings

HEADING_PATTERN = re.compile(
    r"^\s{0,3}#{1,6}\s+.+$",
    re.MULTILINE,
)


# ==========================================================
# Hybrid Chunker
# ==========================================================

class HybridChunker(BaseChunker):
    """
    Fast structure-aware chunker.

    Content is divided into:

        normal text
            ↓
        recursive splitter

        table
            ↓
        table-aware splitting

        code
            ↓
        code-aware splitting

    The chunker does NOT call an LLM.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):

        logger.info(
            "Initializing Hybrid Chunker "
            "(chunk_size=%d, chunk_overlap=%d).",
            chunk_size,
            chunk_overlap,
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = min(
            chunk_overlap,
            max(0, chunk_size // 3),
        )

        # --------------------------------------------------
        # Normal text splitter
        # --------------------------------------------------

        self.text_splitter = (
            RecursiveCharacterTextSplitter(

                chunk_size=self.chunk_size,

                chunk_overlap=self.chunk_overlap,

                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    "? ",
                    "! ",
                    "; ",
                    ", ",
                    " ",
                    "",
                ],

                keep_separator=True,

            )
        )

    # ======================================================
    # Public API
    # ======================================================

    def chunk(
        self,
        documents: List[Document],
    ) -> List[Document]:

        if not documents:

            return []

        start = time.perf_counter()

        logger.info(
            "Chunking %d document(s).",
            len(documents),
        )

        all_chunks: List[Document] = []

        for document in documents:

            document_chunks = self._chunk_document(
                document
            )

            all_chunks.extend(
                document_chunks
            )

        # --------------------------------------------------
        # Add chunk metadata
        # --------------------------------------------------

        all_chunks = self._add_chunk_metadata(
            all_chunks
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        logger.info(
            "Chunking finished in %.3f sec.",
            elapsed,
        )

        logger.info(
            "Created %d chunk(s).",
            len(all_chunks),
        )

        return all_chunks

    # ======================================================
    # Document Chunking
    # ======================================================

    def _chunk_document(
        self,
        document: Document,
    ) -> List[Document]:

        text = document.page_content or ""

        if not text.strip():

            return []

        # --------------------------------------------------
        # If Unstructured explicitly identified the element
        # --------------------------------------------------

        category = str(
            document.metadata.get(
                "category",
                "",
            )
        ).lower()

        if category in {
            "table",
        }:

            return self._chunk_table(
                document
            )

        if category in {
            "code",
            "codeblock",
        }:

            return self._chunk_code(
                document
            )

        # --------------------------------------------------
        # Detect mixed Markdown content
        # --------------------------------------------------

        blocks = self._extract_structured_blocks(
            text
        )

        # No special structures
        # → normal fast recursive chunking

        if not blocks:

            return self._chunk_normal_text(
                document
            )

        # --------------------------------------------------
        # Process mixed document
        # --------------------------------------------------

        chunks: List[Document] = []

        for block_type, content in blocks:

            if not content.strip():

                continue

            if block_type == "code":

                chunks.extend(
                    self._chunk_code(
                        self._copy_document(
                            document,
                            content,
                        )
                    )
                )

            elif block_type == "table":

                chunks.extend(
                    self._chunk_table(
                        self._copy_document(
                            document,
                            content,
                        )
                    )
                )

            elif block_type == "heading":

                # Heading is processed together with its
                # following text by normal processing.

                chunks.extend(
                    self._chunk_normal_text(
                        self._copy_document(
                            document,
                            content,
                        )
                    )
                )

            else:

                chunks.extend(
                    self._chunk_normal_text(
                        self._copy_document(
                            document,
                            content,
                        )
                    )
                )

        return chunks

    # ======================================================
    # Structured Block Detection
    # ======================================================

    @staticmethod
    def _extract_structured_blocks(
        text: str,
    ) -> List[tuple[str, str]]:

        """
        Break a Markdown document into semantic blocks.

        Supported:

        - code blocks
        - tables
        - headings
        - normal text
        """

        lines = text.splitlines(
            keepends=True
        )

        blocks = []

        current_type = "text"
        current_lines: List[str] = []

        inside_code = False

        def flush():

            nonlocal current_lines

            if current_lines:

                content = "".join(
                    current_lines
                )

                if content.strip():

                    blocks.append(
                        (
                            current_type,
                            content,
                        )
                    )

                current_lines = []

        i = 0

        while i < len(lines):

            line = lines[i]

            # --------------------------------------------------
            # Code block
            # --------------------------------------------------

            if line.strip().startswith("```"):

                flush()

                code_lines = [
                    line
                ]

                i += 1

                while i < len(lines):

                    code_lines.append(
                        lines[i]
                    )

                    if lines[i].strip().startswith(
                        "```"
                    ):

                        i += 1
                        break

                    i += 1

                blocks.append(
                    (
                        "code",
                        "".join(code_lines),
                    )
                )

                continue

            # --------------------------------------------------
            # Markdown table
            # --------------------------------------------------

            if (
                TABLE_LINE_PATTERN.match(line)
                and i + 1 < len(lines)
                and TABLE_SEPARATOR_PATTERN.match(
                    lines[i + 1]
                )
            ):

                flush()

                table_lines = [
                    line,
                    lines[i + 1],
                ]

                i += 2

                while i < len(lines):

                    if not TABLE_LINE_PATTERN.match(
                        lines[i]
                    ):

                        break

                    table_lines.append(
                        lines[i]
                    )

                    i += 1

                blocks.append(
                    (
                        "table",
                        "".join(table_lines),
                    )
                )

                continue

            # --------------------------------------------------
            # Heading
            # --------------------------------------------------

            if HEADING_PATTERN.match(line):

                flush()

                blocks.append(
                    (
                        "heading",
                        line,
                    )
                )

                i += 1

                continue

            # --------------------------------------------------
            # Normal text
            # --------------------------------------------------

            current_type = "text"

            current_lines.append(
                line
            )

            i += 1

        flush()

        return blocks

    # ======================================================
    # Normal Text
    # ======================================================

    def _chunk_normal_text(
        self,
        document: Document,
    ) -> List[Document]:

        text = document.page_content.strip()

        if not text:

            return []

        split_documents = (
            self.text_splitter.split_documents(
                [document]
            )
        )

        for chunk in split_documents:

            chunk.metadata = (
                chunk.metadata.copy()
            )

            chunk.metadata[
                "chunk_type"
            ] = "text"

        return split_documents

    # ======================================================
    # Table Chunking
    # ======================================================

    def _chunk_table(
        self,
        document: Document,
    ) -> List[Document]:

        """
        Preserve table structure.

        Small tables remain intact.

        Large tables are split by rows instead
        of arbitrary characters.
        """

        text = document.page_content.strip()

        if not text:

            return []

        lines = [
            line
            for line in text.splitlines()
            if line.strip()
        ]

        # --------------------------------------------------
        # Small table
        # --------------------------------------------------

        if len(text) <= self.chunk_size:

            chunk = self._copy_document(
                document,
                text,
            )

            chunk.metadata[
                "chunk_type"
            ] = "table"

            return [chunk]

        # --------------------------------------------------
        # Large table
        # --------------------------------------------------

        chunks = []

        header = []

        if len(lines) >= 2:

            header = lines[:2]

        current_rows = list(header)

        current_length = sum(
            len(row)
            for row in current_rows
        )

        for row in lines[2:]:

            row_length = len(row) + 1

            if (
                current_rows
                and current_length
                + row_length
                > self.chunk_size
            ):

                table_text = "\n".join(
                    current_rows
                )

                chunks.append(
                    self._copy_document(
                        document,
                        table_text,
                    )
                )

                # Start next chunk with header
                current_rows = list(header)

                current_length = sum(
                    len(row)
                    for row in current_rows
                )

            current_rows.append(row)

            current_length += row_length

        if current_rows:

            table_text = "\n".join(
                current_rows
            )

            chunks.append(
                self._copy_document(
                    document,
                    table_text,
                )
            )

        for chunk in chunks:

            chunk.metadata[
                "chunk_type"
            ] = "table"

        return chunks

    # ======================================================
    # Code Chunking
    # ======================================================

    def _chunk_code(
        self,
        document: Document,
    ) -> List[Document]:

        """
        Preserve fenced code blocks.

        Small blocks remain intact.

        Large blocks are split by lines.
        """

        text = document.page_content.strip()

        if not text:

            return []

        # --------------------------------------------------
        # Small code block
        # --------------------------------------------------

        if len(text) <= self.chunk_size:

            chunk = self._copy_document(
                document,
                text,
            )

            chunk.metadata[
                "chunk_type"
            ] = "code"

            return [chunk]

        # --------------------------------------------------
        # Large code block
        # --------------------------------------------------

        lines = text.splitlines()

        chunks = []

        current_lines = []

        current_length = 0

        for line in lines:

            line_length = len(line) + 1

            if (
                current_lines
                and current_length
                + line_length
                > self.chunk_size
            ):

                code_text = "\n".join(
                    current_lines
                )

                chunks.append(
                    self._copy_document(
                        document,
                        code_text,
                    )
                )

                # Lightweight line overlap
                overlap_lines = (
                    self._get_code_overlap(
                        current_lines
                    )
                )

                current_lines = (
                    overlap_lines
                )

                current_length = sum(
                    len(item) + 1
                    for item in current_lines
                )

            current_lines.append(
                line
            )

            current_length += line_length

        if current_lines:

            code_text = "\n".join(
                current_lines
            )

            chunks.append(
                self._copy_document(
                    document,
                    code_text,
                )
            )

        for chunk in chunks:

            chunk.metadata[
                "chunk_type"
            ] = "code"

        return chunks

    # ======================================================
    # Code Overlap
    # ======================================================

    def _get_code_overlap(
        self,
        lines: List[str],
    ) -> List[str]:

        """
        Keep a small number of previous lines
        to preserve local code context.

        This is intentionally lightweight.
        """

        overlap = []

        length = 0

        for line in reversed(lines):

            if (
                length
                + len(line)
                + 1
                > self.chunk_overlap
            ):

                break

            overlap.insert(
                0,
                line,
            )

            length += (
                len(line)
                + 1
            )

        return overlap

    # ======================================================
    # Metadata
    # ======================================================

    @staticmethod
    def _add_chunk_metadata(
        chunks: List[Document],
    ) -> List[Document]:

        page_chunk_counter = defaultdict(
            int
        )

        page_total_counter = defaultdict(
            int
        )

        # --------------------------------------------------
        # Count chunks per page
        # --------------------------------------------------

        for chunk in chunks:

            page = chunk.metadata.get(
                "page",
                -1,
            )

            page_total_counter[
                page
            ] += 1

        # --------------------------------------------------
        # Add metadata
        # --------------------------------------------------

        for chunk in chunks:

            metadata = (
                chunk.metadata.copy()
            )

            page = metadata.get(
                "page",
                -1,
            )

            metadata[
                "chunk_id"
            ] = str(
                uuid.uuid4()
            )

            metadata[
                "chunk_index"
            ] = page_chunk_counter[
                page
            ]

            metadata[
                "total_chunks_in_page"
            ] = page_total_counter[
                page
            ]

            # Preserve OCR information
            metadata.setdefault(
                "ocr",
                False,
            )

            # Preserve chunk type
            metadata.setdefault(
                "chunk_type",
                "text",
            )

            page_chunk_counter[
                page
            ] += 1

            chunk.metadata = metadata

        return chunks

    # ======================================================
    # Utility
    # ======================================================

    @staticmethod
    def _copy_document(
        document: Document,
        content: str,
    ) -> Document:

        return Document(

            page_content=content,

            metadata=(
                document.metadata.copy()
            ),

        )


# ==========================================================
# Public API
# ==========================================================

def chunk_documents(
    documents: List[Document],
) -> List[Document]:

    chunker = HybridChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return chunker.chunk(
        documents
    )


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    from rag.ingestion.cleaner import (
        clean_documents
    )

    from rag.ingestion.loader import (
        load_document
    )

    from rag.ingestion.metadata import (
        enrich_metadata
    )

    from rag.ingestion.parser import (
        parse_documents
    )

    pdf_path = (
        "documents/Must KNOW.pdf"
    )

    logger.info(
        "Starting hybrid chunker test."
    )

    start = time.perf_counter()

    docs = load_document(
        pdf_path
    )

    docs = parse_documents(
        docs
    )

    docs = clean_documents(
        docs
    )

    docs = enrich_metadata(
        docs
    )

    preprocessing_time = (
        time.perf_counter()
        - start
    )

    logger.info(
        "Preprocessing completed "
        "in %.3f sec.",
        preprocessing_time,
    )

    # --------------------------------------------------
    # Chunk
    # --------------------------------------------------

    start = time.perf_counter()

    chunks = chunk_documents(
        docs
    )

    chunking_time = (
        time.perf_counter()
        - start
    )

    logger.info(
        "Chunking completed "
        "in %.3f sec.",
        chunking_time,
    )

    logger.info(
        "Total chunks created: %d",
        len(chunks),
    )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    chunk_types = defaultdict(int)

    for chunk in chunks:

        chunk_types[
            chunk.metadata.get(
                "chunk_type",
                "unknown",
            )
        ] += 1

    logger.info(
        "Chunk types: %s",
        dict(chunk_types),
    )

    # --------------------------------------------------
    # Sample
    # --------------------------------------------------

    if chunks:

        logger.info(
            "First chunk metadata: %s",
            chunks[0].metadata,
        )

        logger.info(
            "First chunk type: %s",
            chunks[0].metadata.get(
                "chunk_type"
            ),
        )

        logger.info(
            "First chunk content:\n%s",
            chunks[0].page_content[:500],
        )

    logger.info(
        "Hybrid chunker test completed."
    )