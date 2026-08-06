"""
prompt_builder.py

Responsible for building the final prompt for the LLM.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from rag.prompts.templates import RAG_TEMPLATE

from utils.logger import get_logger

logger = get_logger(__name__)


class PromptBuilder:

    def __init__(self):

        logger.info("Initializing Prompt Builder")

    def build(
        self,
        query: str,
        documents: List[Document],
    ) -> str:

        logger.info(
            "Building prompt using %d retrieved document(s).",
            len(documents),
        )

        context = self._build_context(documents)

        prompt = RAG_TEMPLATE.format(
            context=context,
            question=query,
        )

        logger.info("Prompt built successfully.")

        return prompt

    def _build_context(
        self,
        documents: List[Document],
    ) -> str:

        context_parts = []

        for index, document in enumerate(documents, start=1):

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page",
                "-"
            )

            context_parts.append(
                f"""
Document {index}

Source : {source}
Page   : {page}

Content:
{document.page_content}
"""
            )

        return "\n".join(context_parts)