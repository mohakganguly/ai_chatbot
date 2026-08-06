"""
Base interface for all chunkers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseChunker(ABC):

    @abstractmethod
    def chunk(
        self,
        documents: list[Document]
    ) -> list[Document]:
        """
        Split documents into chunks.
        """
        pass