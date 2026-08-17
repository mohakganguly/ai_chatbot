# """
# loader.py

# Responsible for loading different document types into LangChain Documents.
# """

# from __future__ import annotations

# from abc import ABC, abstractmethod
# from pathlib import Path
# from typing import List

# from langchain_core.documents import Document
# from langchain_community.document_loaders import (
#     PyPDFLoader,
#     TextLoader,
#     Docx2txtLoader,
#     UnstructuredMarkdownLoader,
#     UnstructuredHTMLLoader,
# )

# from utils.logger import get_logger

# logger = get_logger(__name__)


# # ==========================================================
# # Base Loader
# # ==========================================================

# class BaseDocumentLoader(ABC):

#     @abstractmethod
#     def load(self, file_path: str) -> List[Document]:
#         """Load a document and return LangChain Documents."""
#         pass


# # ==========================================================
# # PDF Loader
# # ==========================================================

# class PDFLoader(BaseDocumentLoader):

#     def load(self, file_path: str) -> List[Document]:

#         loader = PyPDFLoader(file_path)

#         return loader.load()


# # ==========================================================
# # TXT Loader
# # ==========================================================

# class TXTLoader(BaseDocumentLoader):

#     def load(self, file_path: str) -> List[Document]:

#         loader = TextLoader(file_path)

#         return loader.load()


# # ==========================================================
# # DOCX Loader
# # ==========================================================

# class DOCXLoader(BaseDocumentLoader):

#     def load(self, file_path: str) -> List[Document]:

#         loader = Docx2txtLoader(file_path)

#         return loader.load()


# # ==========================================================
# # Markdown Loader
# # ==========================================================

# class MarkdownLoader(BaseDocumentLoader):

#     def load(self, file_path: str) -> List[Document]:

#         loader = UnstructuredMarkdownLoader(file_path)

#         return loader.load()


# # ==========================================================
# # HTML Loader
# # ==========================================================

# class HTMLLoader(BaseDocumentLoader):

#     def load(self, file_path: str) -> List[Document]:

#         loader = UnstructuredHTMLLoader(file_path)

#         return loader.load()


# # ==========================================================
# # Factory
# # ==========================================================

# class DocumentLoaderFactory:
#     """
#     Factory responsible for selecting the correct loader
#     based on the file extension.
#     """

#     _LOADERS = {
#         ".pdf": PDFLoader,
#         ".txt": TXTLoader,
#         ".docx": DOCXLoader,
#         ".md": MarkdownLoader,
#         ".markdown": MarkdownLoader,
#         ".html": HTMLLoader,
#         ".htm": HTMLLoader,
#     }

#     @classmethod
#     def create_loader(cls, file_path: str) -> BaseDocumentLoader:

#         extension = Path(file_path).suffix.lower()

#         loader_cls = cls._LOADERS.get(extension)

#         if loader_cls is None:

#             supported = ", ".join(cls._LOADERS.keys())

#             logger.error(
#                 "Unsupported file type '%s'. Supported: %s",
#                 extension,
#                 supported,
#             )

#             raise ValueError(
#                 f"Unsupported file type '{extension}'. "
#                 f"Supported types: {supported}"
#             )

#         logger.info("Using %s for '%s'", loader_cls.__name__, file_path)

#         return loader_cls()


# # ==========================================================
# # Public API
# # ==========================================================

# def load_document(file_path: str) -> List[Document]:
#     """
#     Load a document from disk.

#     Parameters
#     ----------
#     file_path : str

#     Returns
#     -------
#     List[Document]
#     """

#     path = Path(file_path)

#     if not path.exists():

#         logger.error("File not found: %s", file_path)

#         raise FileNotFoundError(
#             f"File not found: {file_path}"
#         )

#     try:

#         logger.info("Loading document: %s", file_path)

#         loader = DocumentLoaderFactory.create_loader(file_path)

#         documents = loader.load(file_path)

#         if not documents:

#             logger.warning(
#                 "No content found in '%s'",
#                 file_path
#             )

#             raise ValueError(
#                 "No content found in the document."
#             )

#         logger.info(
#             "Successfully loaded %d document(s)",
#             len(documents)
#         )

#         return documents

#     except Exception:

#         logger.exception(
#             "Failed to load document: %s",
#             file_path
#         )

#         raise


# # ==========================================================
# # Manual Testing
# # ==========================================================

# if __name__ == "__main__":

#     FILE_PATH = "documents/Must KNOW.pdf"

#     docs = load_document(FILE_PATH)

#     logger.info("========== Loader Test ==========")

#     logger.info("Pages Loaded: %d", len(docs))

#     logger.info("Metadata:\n%s", docs[0].metadata)

#     logger.info("Content Preview:\n%s", docs[0].page_content[:1000])


"""
loader.py

Enterprise Document Loader

features
-----
1. Extremely fast extraction.
2. Markdown-preserving PDF parsing.
3. Selective OCR only when needed.
4. Parallel OCR.
5. Lazy imports.
6. Automatic fallback chain.
7. Compatible with the remaining ingestion pipeline.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List
import pymupdf
from langchain_core.documents import Document

from utils.logger import get_logger

logger = get_logger(__name__)

class BaseDocumentLoader(ABC):
    """
    Base interface for every document loader.
    """

    @abstractmethod
    def load(
        self,
        file_path: str,
    ) -> List[Document]:
        pass

class StageTimer:

    def __init__(
        self,
        stage: str,
    ):
        self.stage = stage

    def __enter__(self):

        self.start = time.perf_counter()

        logger.info(
            "Starting %s",
            self.stage,
        )

        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ):

        elapsed = (
            time.perf_counter()
            - self.start
        )

        logger.info(
            "%s finished in %.3f sec",
            self.stage,
            elapsed,
        )

class OCRProcessor:
    """
    Performs OCR only on pages that
    contain insufficient text.
    """

    def __init__(

        self,

        dpi: int = 200,

        threshold: int = 40,

        workers: int = 4,

    ):

        self.dpi = dpi

        self.threshold = threshold

        self.workers = workers

    def _ocr_page(

        self,

        pdf,

        page_index: int,

    ):

        import pytesseract
        from PIL import Image

        page = pdf[page_index]

        pix = page.get_pixmap(
            dpi=self.dpi,
        )

        image = Image.frombytes(

            "RGB",

            (pix.width, pix.height),

            pix.samples,

        )

        text = pytesseract.image_to_string(
            image
        )

        return page_index, text

    def process(

        self,

        pdf,

        documents: List[Document],

    ) -> List[Document]:

            pages_to_ocr = [

                i

                for i, doc in enumerate(documents)

                if len(doc.page_content.strip())
                < self.threshold

            ]

            if not pages_to_ocr:

                logger.info(
                    "OCR not required."
                )

                return documents

            logger.info(

                "Running OCR on %d page(s)",

                len(pages_to_ocr),

            )

            with ThreadPoolExecutor(

                max_workers=self.workers,

            ) as executor:

                futures = [

                    executor.submit(

                        self._ocr_page,

                        pdf,

                        page,

                    )

                    for page in pages_to_ocr

                ]

                for future in futures:

                    page_index, text = future.result()

                    documents[
                        page_index
                    ].page_content = text

                    documents[
                        page_index
                    ].metadata["ocr"] = True

            return documents


def build_metadata(

    source: str,

    page: int | None,

    loader: str,

    **extra,

):

    metadata = {

        "source": source,

        "page": page,

        "loader": loader,

        "ocr": False,

    }

    metadata.update(extra)

    return metadata

# ==========================================================
# Enterprise PDF Loader
# ==========================================================

class PDFLoader(BaseDocumentLoader):
    """
    High-performance PDF loader.

    Pipeline

    PyMuPDF4LLM Markdown Extraction
            ↓
    Selective OCR
            ↓
    Return Documents

    OCR only runs on pages with
    almost no extracted text.
    """

    def __init__(

        self,

        use_markdown: bool = True,

        ocr_threshold: int = 20,

        enable_ocr: bool = True,

        min_text_length: int = 300,

    ):

        self.use_markdown = use_markdown

        self.enable_ocr = enable_ocr

        self.ocr = OCRProcessor(
            threshold=ocr_threshold,
        )

        self.min_text_length = min_text_length

    def load(
        self,
        file_path: str,
    ) -> List[Document]:

        logger.info("Loading PDF '%s'", file_path)

        docs = self._extract_fast_text(file_path)

        total_chars = sum(
            len(doc.page_content.strip())
            for doc in docs
        )

        logger.info(
            "Fast extraction produced %d characters",
            total_chars,
        )

        if total_chars >= self.min_text_length:

            logger.info(
                "Fast extraction successful."
            )

            return docs

        logger.info(
            "Switching to Markdown extraction."
        )

        docs = self._extract_markdown(file_path)

        if not self.enable_ocr:

            return docs

        need_ocr = any(
            len(doc.page_content.strip()) < self.ocr_threshold
            for doc in docs
        )

        if not need_ocr:

            logger.info("OCR not required.")

            return docs

        logger.info("Running OCR.")

        return self._run_ocr_if_needed(
            file_path,
            docs,
        )

    def _extract_fast_text(
        self,
        file_path: str,
    ) -> List[Document]:

        logger.info("Running fast PyMuPDF extraction")

        pdf = pymupdf.open(file_path)

        metadata = pdf.metadata

        page_count = pdf.page_count

        docs = []

        for page_no, page in enumerate(pdf, start=1):

            docs.append(

                Document(

                    page_content=page.get_text("text"),

                    metadata={

                        "source": file_path,

                        "page": page_no,

                        "loader": "pymupdf",

                        "ocr": False,

                        "file_path": file_path,

                        "page_count": page_count,

                        **metadata,

                    },

                )

            )

        pdf.close()

        return docs

    def _extract_markdown(
        self,
        file_path: str,
    ) -> List[Document]:

        import pymupdf4llm

        with StageTimer(
            "Markdown Extraction"
        ):

            pages = pymupdf4llm.to_markdown(

                file_path,

                page_chunks=True,

            )

        documents = []

        for index, page in enumerate(pages):

            documents.append(

                Document(

                    page_content=page["text"],

                    metadata=build_metadata(

                        source=file_path,

                        page=index + 1,

                        loader="pymupdf4llm",

                        **page.get(
                            "metadata",
                            {},
                        ),

                    ),

                )

            )

        return documents

    def _load_text(
        self,
        file_path: str,
    ) -> List[Document]:

        

        documents = []

        with StageTimer(
            "Text Extraction"
        ):

            with pymupdf.open(
                file_path
            ) as pdf:

                for page_number, page in enumerate(

                    pdf,

                    start=1,

                ):

                    documents.append(

                        Document(

                            page_content=page.get_text(),

                            metadata=build_metadata(

                                source=file_path,

                                page=page_number,

                                loader="pymupdf",

                            ),

                        )

                    )

        return documents

    def _run_ocr_if_needed(

        self,

        file_path: str,

        documents: List[Document],

    ) -> List[Document]:

        import pymupdf

        with StageTimer("OCR"):

            with pymupdf.open(
                file_path
            ) as pdf:

                documents = self.ocr.process(

                    pdf,

                    documents,

                )

        return documents

    @staticmethod
    def has_content(
        documents: List[Document],
    ) -> bool:

        if not documents:

            return False

        for document in documents:

            if document.page_content.strip():

                return True

        return False

# ==========================================================
# TXT Loader
# ==========================================================

class TXTLoader(BaseDocumentLoader):

    def load(
        self,
        file_path: str,
    ) -> List[Document]:

        from langchain_community.document_loaders import TextLoader

        with StageTimer("TXT Loading"):

            docs = TextLoader(
                file_path,
                encoding="utf-8",
            ).load()

        for doc in docs:

            doc.metadata = build_metadata(

                source=file_path,

                page=1,

                loader="TextLoader",

            )

        return docs

# ==========================================================
# DOCX Loader
# ==========================================================

class DOCXLoader(BaseDocumentLoader):

    def load(
        self,
        file_path: str,
    ) -> List[Document]:

        from langchain_community.document_loaders import (
            Docx2txtLoader,
        )

        with StageTimer("DOCX Loading"):

            docs = Docx2txtLoader(
                file_path
            ).load()

        for doc in docs:

            doc.metadata = build_metadata(

                source=file_path,

                page=1,

                loader="Docx2txt",

            )

        return docs

# ==========================================================
# Markdown Loader
# ==========================================================

class MarkdownLoader(BaseDocumentLoader):

    def load(
        self,
        file_path: str,
    ) -> List[Document]:

        with StageTimer("Markdown Loading"):

            with open(

                file_path,

                "r",

                encoding="utf-8",

            ) as f:

                text = f.read()

        return [

            Document(

                page_content=text,

                metadata=build_metadata(

                    source=file_path,

                    page=1,

                    loader="Markdown",

                ),

            )

        ]

# ==========================================================
# HTML Loader
# ==========================================================

class HTMLLoader(BaseDocumentLoader):

    def load(
        self,
        file_path: str,
    ) -> List[Document]:

        from bs4 import BeautifulSoup

        with StageTimer("HTML Loading"):

            with open(

                file_path,

                encoding="utf-8",

            ) as f:

                html = f.read()

        soup = BeautifulSoup(

            html,

            "html.parser",

        )

        text = soup.get_text(

            separator="\n",

        )

        return [

            Document(

                page_content=text,

                metadata=build_metadata(

                    source=file_path,

                    page=1,

                    loader="HTML",

                ),

            )

        ]

# ==========================================================
# Image Loader
# ==========================================================

class ImageLoader(BaseDocumentLoader):

    def load(
        self,
        file_path: str,
    ) -> List[Document]:

        import pytesseract

        from PIL import Image

        with StageTimer("Image OCR"):

            image = Image.open(
                file_path
            )

            text = pytesseract.image_to_string(
                image
            )

        return [

            Document(

                page_content=text,

                metadata=build_metadata(

                    source=file_path,

                    page=1,

                    loader="ImageOCR",

                    ocr=True,

                ),

            )

        ]




# ==========================================================
# Unstructured Loader
# ==========================================================

class UnstructuredLoader(BaseDocumentLoader):

    def __init__(

        self,

        strategy="hi_res",

    ):

        self.strategy = strategy

    def load(

        self,

        file_path: str,

    ):

        from unstructured.partition.auto import partition

        with StageTimer(

            f"Unstructured ({self.strategy})"

        ):

            elements = partition(

                filename=file_path,

                strategy=self.strategy,

            )

        docs = []

        for element in elements:

            metadata = build_metadata(

                source=file_path,

                page=getattr(

                    getattr(

                        element,

                        "metadata",

                        None,

                    ),

                    "page_number",

                    None,

                ),

                loader="Unstructured",

                category=getattr(

                    element,

                    "category",

                    None,

                ),

            )

            docs.append(

                Document(

                    page_content=str(element),

                    metadata=metadata,

                )

            )

        return docs
    
# ==========================================================
# LlamaParse Loader
# ==========================================================

class LlamaParseLoader(BaseDocumentLoader):

    def __init__(

        self,

        api_key=None,

        result_type="markdown",

    ):

        from llama_parse import LlamaParse

        self.parser = LlamaParse(

            api_key=api_key

            or os.environ[

                "LLAMA_CLOUD_API_KEY"

            ],

            result_type=result_type,

        )

    def load(

        self,

        file_path: str,

    ):

        with StageTimer(

            "LlamaParse"

        ):

            result = self.parser.load_data(

                file_path

            )

        docs = []

        for page in result:

            docs.append(

                Document(

                    page_content=page.text,

                    metadata=build_metadata(

                        source=file_path,

                        page=page.metadata.get(

                            "page",

                            None,

                        ),

                        loader="LlamaParse",

                        **page.metadata,

                    ),

                )

            )

        return docs
    
# ==========================================================
# Loader Factory
# ==========================================================

class DocumentLoaderFactory:

    _instances = {}

    _mapping = {

        ".pdf": PDFLoader,

        ".txt": TXTLoader,

        ".docx": DOCXLoader,

        ".md": MarkdownLoader,

        ".markdown": MarkdownLoader,

        ".html": HTMLLoader,

        ".htm": HTMLLoader,

        ".png": ImageLoader,

        ".jpg": ImageLoader,

        ".jpeg": ImageLoader,

        ".bmp": ImageLoader,

        ".tiff": ImageLoader,

    }

    @classmethod
    def get_loader(

        cls,

        extension: str,

    ) -> BaseDocumentLoader:

        extension = extension.lower()

        if extension not in cls._mapping:

            raise ValueError(

                f"Unsupported file type '{extension}'"

            )

        if extension not in cls._instances:

            cls._instances[extension] = (

                cls._mapping[extension]()

            )

        return cls._instances[extension]

# ==========================================================
# Validation
# ==========================================================

def _has_content(

    documents: List[Document],

) -> bool:

    if not documents:

        return False

    return any(

        doc.page_content.strip()

        for doc in documents

    )

# ==========================================================
# PDF Fallback Chain
# ==========================================================

def _fallback_pdf(

    file_path: str,

) -> List[Document]:

    logger.warning(

        "Primary PDF extraction failed."

    )

    # --------------------------------------------
    # Try Unstructured
    # --------------------------------------------

    try:

        logger.info(

            "Trying Unstructured fallback..."

        )

        loader = UnstructuredLoader()

        docs = loader.load(file_path)

        if _has_content(docs):

            return docs

    except Exception:

        logger.exception(

            "Unstructured failed."

        )

    # --------------------------------------------
    # Optional LlamaParse
    # --------------------------------------------

    if os.getenv(

        "ENABLE_LLAMA_PARSE",

        "false",

    ).lower() == "true":

        try:

            logger.info(

                "Trying LlamaParse..."

            )

            loader = LlamaParseLoader()

            docs = loader.load(file_path)

            if _has_content(docs):

                return docs

        except Exception:

            logger.exception(

                "LlamaParse failed."

            )

    raise RuntimeError(

        "No loader could extract text."

    )

# ==========================================================
# Public API
# ==========================================================

def load_document(

    file_path: str,

) -> List[Document]:

    path = Path(file_path)

    if not path.exists():

        raise FileNotFoundError(

            file_path

        )

    extension = (

        path.suffix.lower()

    )

    logger.info(

        "=" * 80

    )

    logger.info(

        "Loading %s",

        path.name,

    )

    logger.info(

        "Detected type : %s",

        extension,

    )

    with StageTimer(

        "Document Loading"

    ):

        try:

            loader = (

                DocumentLoaderFactory.get_loader(

                    extension

                )

            )

            documents = loader.load(

                file_path

            )

            if not _has_content(

                documents

            ):

                raise RuntimeError(

                    "Empty extraction"

                )

            logger.info(

                "Successfully loaded %d document(s).",

                len(documents),

            )

            return documents

        except Exception as exc:

            logger.warning(

                "Primary loader failed: %s",

                exc,

            )

            if extension == ".pdf":

                return _fallback_pdf(

                    file_path

                )

            raise

# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    pdf = "documents/sample.pdf"

    docs = load_document(

        pdf,

    )

    print()

    print("=" * 80)

    print(

        "Pages:",

        len(docs),

    )

    print("=" * 80)

    print()

    for page in docs[:3]:

        print(

            page.metadata

        )

        print()

        print(

            page.page_content[:500]

        )

        print()

        print("-" * 80)