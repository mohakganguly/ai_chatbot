"""
query_rewriter.py

Enterprise Query Rewriter.

Responsible for:
1. Rewriting vague queries
2. Expanding abbreviations
3. Improving retrieval quality
"""

from __future__ import annotations

import time

from core.llm import rewriter_llm

from rag.prompts.query_rewriter_template import (
    QUERY_REWRITER_TEMPLATE,
)

from rag.prompts.history_query_rewriter_template import (
    HISTORY_QUERY_REWRITER_TEMPLATE,
)

from utils.logger import get_logger

logger = get_logger(__name__)


class QueryRewriter:

    SQL_KEYWORDS = {
        "SELECT",
        "INSERT",
        "UPDATE",
        "DELETE",
        "CREATE",
        "DROP",
        "ALTER",
        "WHERE",
        "FROM",
        "JOIN",
        "GROUP BY",
        "ORDER BY",
    }

    def __init__(self):

        logger.info(
            "Initializing Query Rewriter"
        )

        self.llm = rewriter_llm

    def rewrite(
        self,
        query: str,
        history: str = "",
    ) -> str:

        logger.info(
            "Rewriting query..."
        )

        logger.info(
            "Original Query : %s",
            query,
        )

        start = time.perf_counter()

        try:

            if history.strip():

                prompt = HISTORY_QUERY_REWRITER_TEMPLATE.format(
                    history=history,
                    query=query,
                )

            else:

                prompt = QUERY_REWRITER_TEMPLATE.format(
                    query=query,
                )

            response = self.llm.invoke(
                prompt
            )

            rewritten_query = response.content.strip()

            if not self._is_valid_rewrite(
                original=query,
                rewritten=rewritten_query,
            ):

                logger.warning(
                    "Invalid rewritten query detected. "
                    "Using original query."
                )

                rewritten_query = query

            elapsed = (
                time.perf_counter()
                - start
            )

            logger.info(
                "Rewritten Query : %s",
                rewritten_query,
            )

            logger.info(
                "Query rewritten in %.3f sec.",
                elapsed,
            )

            return rewritten_query

        except Exception:

            logger.exception(
                "Query rewriting failed."
            )

            logger.info(
                "Using original query."
            )

            return query

    def _is_valid_rewrite(
        self,
        original: str,
        rewritten: str,
    ) -> bool:
        """
        Validate the rewritten query before
        sending it to the retriever.
        """

        rewritten_upper = rewritten.upper()

        # Reject SQL

        if any(
            keyword in rewritten_upper
            for keyword in self.SQL_KEYWORDS
        ):

            logger.warning(
                "SQL detected in rewritten query."
            )

            return False

        # Reject markdown code blocks

        if rewritten.startswith("```"):

            logger.warning(
                "Code block detected in rewritten query."
            )

            return False

        # Reject JSON

        if rewritten.startswith("{"):

            logger.warning(
                "JSON detected in rewritten query."
            )

            return False

        # Reject XML

        if rewritten.startswith("<"):

            logger.warning(
                "XML detected in rewritten query."
            )

            return False

        # Reject empty output

        if not rewritten.strip():

            logger.warning(
                "Empty rewritten query."
            )

            return False

        # Reject extremely long rewrites

        if len(rewritten) > 500:

            logger.warning(
                "Rewritten query too long."
            )

            return False

        return True


if __name__ == "__main__":

    rewriter = QueryRewriter()

    query = "How is it different?"

    history = """
User: Explain Git Merge.
Assistant: Git Merge combines two branches.
"""

    rewritten = rewriter.rewrite(
        query=query,
        history=history,
    )

    print("=" * 80)
    print("Original")
    print(query)

    print()

    print("Rewritten")
    print(rewritten)