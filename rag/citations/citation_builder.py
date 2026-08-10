"""
citation_builder.py

Responsible for building citations from
retrieved documents.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from utils.logger import get_logger


logger = get_logger(__name__)


class CitationBuilder:

    def __init__(self):

        logger.info(
            "Initializing Citation Builder"
        )

    def build(
        self,
        documents: List[Document],
    ) -> List[dict]:

        logger.info(
            "Building citations from %d document(s).",
            len(documents),
        )

        citations = []

        seen = set()

        for document in documents:

            source = document.metadata.get(
                "source",
                "Unknown",
            )

            page = document.metadata.get(
                "page",
                "-",
            )

            key = (source, page)

            if key in seen:
                continue

            seen.add(key)

            citations.append(
                {
                    "source": source,
                    "page": page,
                }
            )

        logger.info(
            "Generated %d citation(s).",
            len(citations),
        )

        return citations