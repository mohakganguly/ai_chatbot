"""
retriever_tool.py
"""

from __future__ import annotations

from tools.base import (
    BaseTool,
    ToolResult,
    ToolContext
)


from rag.services.retrieval_service import (
    RetrievalService,
)

from utils.logger import get_logger

logger = get_logger(__name__)


class RetrieverTool(BaseTool):

    name = "retriever"

    description = (
        "Search uploaded enterprise documents."
    )

    def __init__(self):

        self.service = RetrievalService()

    def invoke(
        self,
        context: ToolContext
    ) -> ToolResult:

        logger.info("Running Retriever Tool")

        try:

            result = self.service.retrieve(

                query=context.query,

                messages=context.messages,

                current_documents=context.metadata.get(
                    "current_documents",
                    [],
                ),
            )
            extracted_metadata = {}
            if isinstance(result, dict):
                extracted_metadata = result.get("metadata", {})
            else:
                extracted_metadata = getattr(result, "metadata", {})

            # Build a dynamic summary of what was found
            item_count = len(result) if hasattr(result, "__len__") else 1
            summary_msg = f"Successfully retrieved {item_count} relevant document segments."
            return ToolResult(
                tool_name=self.name,
                success=True,
                summary=summary_msg,
                output=result,

                metadata=extracted_metadata,
            )

        except Exception as exc:

            logger.exception(
                "Retriever Tool Failed"
            )

            return ToolResult(

                tool_name="retriever_tool",  # Required field fulfilled
                success=False,
                summary="The data retrieval process failed due to an internal structural error.",
                # Required field fulfilled
                error=str(exc),
                output=None
            )