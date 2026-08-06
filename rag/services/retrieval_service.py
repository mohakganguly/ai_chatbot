"""
retrieval_service.py

Enterprise Retrieval Service.

Responsible for the complete
document retrieval pipeline.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage

from graph.history import build_history

from rag.retrieval.retriever import Retriever
from rag.retrieval.query_rewriter import QueryRewriter
from rag.retrieval.rewrite_decider import RewriteDecider
from rag.retrieval.reranker import Reranker

from rag.filtering.context_filter import ContextFilter
from rag.citations.citation_builder import CitationBuilder

from rag.services.schemas import RetrievalResult

from utils.logger import get_logger
from observability.tracing import trace_node

logger = get_logger(__name__)


class RetrievalService:
    """
    Enterprise retrieval pipeline.

    Encapsulates the entire retrieval workflow
    independently of LangGraph.
    """

    def __init__(self):

        self.retriever = Retriever()

        self.rewriter = QueryRewriter()

        self.rewrite_decider = RewriteDecider()

        self.reranker = Reranker()

        self.context_filter = ContextFilter()

        self.citation_builder = CitationBuilder()

    def retrieve(
        self,
        query: str,
        messages: list[BaseMessage],
        current_documents: list[str] | None = None,
    ) -> RetrievalResult:
        """
        Execute the enterprise retrieval pipeline.
        """

        with trace_node(
            "retrieval",
            query=query,
        ):

            logger.info(
                "Starting Retrieval Pipeline"
            )

            # ======================================================
            # History + Rewrite Decision
            # ======================================================

            with trace_node("retrieval.history"):

                history = build_history(
                    messages
                )

                should_rewrite = False

                if history.strip():

                    should_rewrite = (
                        self.rewrite_decider.should_rewrite(
                            query
                        )
                    )

                logger.info(
                    "Rewrite Required : %s",
                    should_rewrite,
                )

            # ======================================================
            # Query Processing
            # ======================================================

            with trace_node(
                "retrieval.query_processing",
                rewrite=should_rewrite,
            ):

                retrieval_query = query

                if should_rewrite:

                    retrieval_query = (
                        self.rewriter.rewrite(
                            query=query,
                            history=history,
                        )
                    )

                logger.info(
                    "Retrieval Query : %s",
                    retrieval_query,
                )

            # ======================================================
            # Search
            # ======================================================

            with trace_node(
                "retrieval.search",
                retrieval_query=retrieval_query,
            ):

                documents = self.retriever.retrieve(
                    retrieval_query
                )

                logger.info(
                    "Current uploaded docs : %s",
                    current_documents,
                )

                for doc in documents:

                    logger.info(
                        "Metadata source=%s | file_name=%s",
                        doc.metadata.get("source"),
                        doc.metadata.get("file_name"),
                    )

                # ------------------------------------------
                # Filter uploaded documents
                # ------------------------------------------

                if current_documents:

                    documents = [

                        doc

                        for doc in documents

                        if doc.metadata.get("source")
                        in current_documents

                    ]

                logger.info(
                    "Documents after file filtering : %d",
                    len(documents),
                )

                logger.info(
                    "Retrieved %d documents",
                    len(documents),
                )

                documents = self.reranker.rerank(

                    query=retrieval_query,

                    documents=documents,
                )

                logger.info(
                    "Documents reranked"
                )

            # ======================================================
            # Post Processing
            # ======================================================

            with trace_node("retrieval.post_processing"):

                documents = (
                    self.context_filter.filter_documents(
                        documents
                    )
                )

                logger.info(
                    "Documents filtered : %d",
                    len(documents),
                )

                citations = (
                    self.citation_builder.build(
                        documents
                    )
                )

            logger.info(
                "Retrieval Pipeline Finished"
            )

            # ======================================================
            # Response
            # ======================================================

            with trace_node(
                "retrieval.response",
                documents=len(documents),
                citations=len(citations),
            ):

                return RetrievalResult(

                    query=query,

                    rewritten_query=retrieval_query,

                    documents=documents,

                    citations=citations,

                )