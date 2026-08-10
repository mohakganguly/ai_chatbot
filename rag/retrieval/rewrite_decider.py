# """
# rewrite_decider.py

# Determines whether the user's query should be rewritten
# using conversation history.
# """

# from __future__ import annotations

# from core.llm import structured_rewrite_llm

# from utils.logger import get_logger

# logger = get_logger(__name__)


# class RewriteDecider:

#     VALID_RESPONSES = {
#         "yes",
#         "no",
#     }

#     def should_rewrite(
#         self,
#         query: str,
#     ) -> bool:

#         prompt = f"""
# You are a classifier.

# Determine whether the user's latest query requires
# conversation history to be understood.

# Examples

# User: Explain Git Merge.
# Answer: NO

# User: How is it different?
# Answer: YES

# User: Explain that again.
# Answer: YES

# User: Compare it with Rebase.
# Answer: YES

# User: According to the uploaded PDF, list Linux topics.
# Answer: NO

# Rules

# - Answer ONLY YES or NO.
# - No explanation.

# Query

# {query}

# Answer
# """

#         try:

#             response = structured_rewrite_llm.invoke(prompt)

#             answer = response.content.strip().lower()

#             logger.info(
#                 "Rewrite Decision -> %s",
#                 response.rewrite,
#             )

#             return response.rewrite

#         except Exception:

#             logger.exception(
#                 "Rewrite decision failed."
#             )

#             return False



# """
# rewrite_decider.py

# Fast deterministic decision about whether a query
# requires conversation history.
# """

# from __future__ import annotations

# import re

# from utils.logger import get_logger


# logger = get_logger(__name__)


# class RewriteDecider:

#     # ---------------------------------------------------------
#     # Queries that commonly depend on previous context
#     # ---------------------------------------------------------

#     CONTEXT_REFERENCES = {
#         "it",
#         "its",
#         "they",
#         "them",
#         "their",
#         "this",
#         "that",
#         "these",
#         "those",
#         "he",
#         "she",
#         "him",
#         "her",
#         "same",
#         "again",
#     }

#     CONTEXT_PHRASES = (
#         "how is it",
#         "how does it",
#         "why is it",
#         "why does it",
#         "what about it",
#         "what about that",
#         "what about this",
#         "explain that again",
#         "explain this again",
#         "tell me more",
#         "compare it",
#         "compare that",
#         "compare this",
#         "difference between them",
#         "difference between these",
#         "same thing",
#         "above",
#         "previous",
#         "earlier",
#     )

#     # ---------------------------------------------------------
#     # Queries that are normally self-contained
#     # ---------------------------------------------------------

#     SELF_CONTAINED_PREFIXES = (
#         "what is",
#         "what are",
#         "who is",
#         "who are",
#         "explain",
#         "define",
#         "how does",
#         "how do",
#         "why does",
#         "why do",
#         "when did",
#         "where is",
#         "where are",
#         "list",
#         "show",
#         "give",
#         "describe",
#     )

#     def should_rewrite(
#         self,
#         query: str,
#     ) -> bool:

#         if not query or not query.strip():

#             return False

#         normalized = " ".join(
#             query.lower().strip().split()
#         )

#         # -----------------------------------------------------
#         # Explicit context phrases
#         # -----------------------------------------------------

#         for phrase in self.CONTEXT_PHRASES:

#             if phrase in normalized:

#                 logger.info(
#                     "Rewrite Decision -> YES "
#                     "(context phrase: '%s')",
#                     phrase,
#                 )

#                 return True

#         # -----------------------------------------------------
#         # Token-level references
#         # -----------------------------------------------------

#         tokens = set(
#             re.findall(
#                 r"\b[\w']+\b",
#                 normalized,
#             )
#         )

#         if tokens.intersection(
#             self.CONTEXT_REFERENCES
#         ):

#             logger.info(
#                 "Rewrite Decision -> YES "
#                 "(context reference detected)."
#             )

#             return True

#         # -----------------------------------------------------
#         # Very short queries are often contextual
#         # -----------------------------------------------------

#         if len(normalized.split()) <= 4:

#             logger.info(
#                 "Rewrite Decision -> YES "
#                 "(short query)."
#             )

#             return True

#         # -----------------------------------------------------
#         # Otherwise treat query as self-contained
#         # -----------------------------------------------------

#         logger.info(
#             "Rewrite Decision -> NO."
#         )

#         return False   


"""
rewrite_decider.py

Fast deterministic decision about whether a query
requires conversation history.

No LLM call is made here.
"""

from __future__ import annotations

import re

from utils.logger import get_logger


logger = get_logger(__name__)


class RewriteDecider:
    """
    Determines whether the latest query depends on
    previous conversation context.

    This is intentionally rule-based because using an LLM
    for every query adds unnecessary latency.
    """

    # ---------------------------------------------------------
    # Explicit context-dependent phrases
    # ---------------------------------------------------------

    CONTEXT_PHRASES = (
        "how is it",
        "how does it",
        "how do they",
        "why is it",
        "why does it",
        "what about it",
        "what about that",
        "what about this",
        "explain that again",
        "explain this again",
        "tell me more",
        "compare it",
        "compare that",
        "compare this",
        "difference between them",
        "difference between these",
        "same thing",
        "same one",
        "the above",
        "above",
        "previous",
        "earlier",
        "before",
    )

    # ---------------------------------------------------------
    # Pronouns / references that commonly require context
    # ---------------------------------------------------------

    CONTEXT_REFERENCES = {
        "it",
        "its",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "same",
        "again",
    }
    SHORT_CONTEXT_QUERIES = {
        "why",
        "how",
        "what about",
        "and then",
        "what next",
        "which one",
        "which is better",
    }
    def should_rewrite(
        self,
        query: str,
    ) -> bool:
        """
        Return True when the query is likely to require
        conversation history.
        """

        if not query or not query.strip():

            logger.info(
                "Rewrite Decision -> NO (empty query)."
            )

            return False

        normalized = re.sub(
            r"[^\w\s']",
            "",
            query.lower(),
        )

        normalized = " ".join(
            normalized.strip().split()
        )
        # -----------------------------------------------------
        # 1. Explicit context phrases
        # -----------------------------------------------------

        for phrase in self.CONTEXT_PHRASES:

            if phrase in normalized:

                logger.info(
                    "Rewrite Decision -> YES "
                    "(context phrase: '%s').",
                    phrase,
                )

                return True

        # -----------------------------------------------------
        # 2. Context references
        # -----------------------------------------------------

        tokens = set(
            re.findall(
                r"\b[\w']+\b",
                normalized,
            )
        )

        if tokens.intersection(
            self.CONTEXT_REFERENCES
        ):

            logger.info(
                "Rewrite Decision -> YES "
                "(context reference detected)."
            )

            return True

        # -----------------------------------------------------
        # 3. Very short queries
        # -----------------------------------------------------

        # Short follow-ups such as:
        #
        #   "Why?"
        #   "How?"
        #   "And?"
        #   "What about it?"
        #
        # are often dependent on previous context.

        if normalized in self.SHORT_CONTEXT_QUERIES:

            logger.info(
                "Rewrite Decision -> YES "
                "(very short query)."
            )

            return True

        # -----------------------------------------------------
        # 4. Otherwise self-contained
        # -----------------------------------------------------

        logger.info(
            "Rewrite Decision -> NO."
        )

        return False


# =============================================================
# Testing
# =============================================================

if __name__ == "__main__":

    decider = RewriteDecider()

    test_queries = [

        # Expected NO
        (
            "What is a Zombie Process?",
            False,
        ),

        (
            "Explain Git Merge.",
            False,
        ),

        (
            "List Linux topics.",
            False,
        ),

        (
            "What is a deadlock?",
            False,
        ),

        # Expected YES
        (
            "How is it different?",
            True,
        ),

        (
            "Explain that again.",
            True,
        ),

        (
            "Compare it with Rebase.",
            True,
        ),

        (
            "Tell me more.",
            True,
        ),

        (
            "Why?",
            True,
        ),

        (
            "What about it?",
            True,
        ),
    ]

    print("=" * 80)
    print("Rewrite Decider Test")
    print("=" * 80)

    passed = 0

    for query, expected in test_queries:

        actual = decider.should_rewrite(
            query
        )

        status = (
            "PASS"
            if actual == expected
            else "FAIL"
        )

        if actual == expected:
            passed += 1

        print(
            f"{status:>5} | "
            f"{query:<35} | "
            f"expected={expected} "
            f"actual={actual}"
        )

    print("=" * 80)

    print(
        f"Passed: {passed}/{len(test_queries)}"
    )