"""
loader.py

Responsible for loading different document types into LangChain Documents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)

from utils.logger import get_logger

logger = get_logger(__name__)


# ==========================================================
# Base Loader
# ==========================================================

class BaseDocumentLoader(ABC):

    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """Load a document and return LangChain Documents."""
        pass


# ==========================================================
# PDF Loader
# ==========================================================

class PDFLoader(BaseDocumentLoader):

    def load(self, file_path: str) -> List[Document]:

        loader = PyPDFLoader(file_path)

        return loader.load()


# ==========================================================
# TXT Loader
# ==========================================================

class TXTLoader(BaseDocumentLoader):

    def load(self, file_path: str) -> List[Document]:

        loader = TextLoader(file_path)

        return loader.load()


# ==========================================================
# DOCX Loader
# ==========================================================

class DOCXLoader(BaseDocumentLoader):

    def load(self, file_path: str) -> List[Document]:

        loader = Docx2txtLoader(file_path)

        return loader.load()


# ==========================================================
# Markdown Loader
# ==========================================================

class MarkdownLoader(BaseDocumentLoader):

    def load(self, file_path: str) -> List[Document]:

        loader = UnstructuredMarkdownLoader(file_path)

        return loader.load()


# ==========================================================
# HTML Loader
# ==========================================================

class HTMLLoader(BaseDocumentLoader):

    def load(self, file_path: str) -> List[Document]:

        loader = UnstructuredHTMLLoader(file_path)

        return loader.load()


# ==========================================================
# Factory
# ==========================================================

class DocumentLoaderFactory:
    """
    Factory responsible for selecting the correct loader
    based on the file extension.
    """

    _LOADERS = {
        ".pdf": PDFLoader,
        ".txt": TXTLoader,
        ".docx": DOCXLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
        ".html": HTMLLoader,
        ".htm": HTMLLoader,
    }

    @classmethod
    def create_loader(cls, file_path: str) -> BaseDocumentLoader:

        extension = Path(file_path).suffix.lower()

        loader_cls = cls._LOADERS.get(extension)

        if loader_cls is None:

            supported = ", ".join(cls._LOADERS.keys())

            logger.error(
                "Unsupported file type '%s'. Supported: %s",
                extension,
                supported,
            )

            raise ValueError(
                f"Unsupported file type '{extension}'. "
                f"Supported types: {supported}"
            )

        logger.info("Using %s for '%s'", loader_cls.__name__, file_path)

        return loader_cls()


# ==========================================================
# Public API
# ==========================================================

def load_document(file_path: str) -> List[Document]:
    """
    Load a document from disk.

    Parameters
    ----------
    file_path : str

    Returns
    -------
    List[Document]
    """

    path = Path(file_path)

    if not path.exists():

        logger.error("File not found: %s", file_path)

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    try:

        logger.info("Loading document: %s", file_path)

        loader = DocumentLoaderFactory.create_loader(file_path)

        documents = loader.load(file_path)

        if not documents:

            logger.warning(
                "No content found in '%s'",
                file_path
            )

            raise ValueError(
                "No content found in the document."
            )

        logger.info(
            "Successfully loaded %d document(s)",
            len(documents)
        )

        return documents

    except Exception:

        logger.exception(
            "Failed to load document: %s",
            file_path
        )

        raise


# ==========================================================
# Manual Testing
# ==========================================================

if __name__ == "__main__":

    FILE_PATH = "documents/Must KNOW.pdf"

    docs = load_document(FILE_PATH)

    logger.info("========== Loader Test ==========")

    logger.info("Pages Loaded: %d", len(docs))

    logger.info("Metadata:\n%s", docs[0].metadata)

    logger.info("Content Preview:\n%s", docs[0].page_content[:1000])