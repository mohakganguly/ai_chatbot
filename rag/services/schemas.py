"""
schemas.py

Shared schemas for retrieval services.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from langchain_core.documents import Document


class RetrievalResult(BaseModel):
    """
    Result produced by the retrieval service.
    """

    query: str = Field(
        description="Original user query."
    )

    rewritten_query: str = Field(
        description="Query actually used for retrieval."
    )

    documents: list[Document] = Field(
        default_factory=list,
        description="Retrieved documents."
    )

    citations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Citation metadata."
    )