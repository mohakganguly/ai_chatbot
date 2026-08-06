"""
rewrite_decider.py

Determines whether the user's query should be rewritten
using conversation history.
"""

from __future__ import annotations

from core.llm import structured_rewrite_llm

from utils.logger import get_logger

logger = get_logger(__name__)


class RewriteDecider:

    VALID_RESPONSES = {
        "yes",
        "no",
    }

    def should_rewrite(
        self,
        query: str,
    ) -> bool:

        prompt = f"""
You are a classifier.

Determine whether the user's latest query requires
conversation history to be understood.

Examples

User: Explain Git Merge.
Answer: NO

User: How is it different?
Answer: YES

User: Explain that again.
Answer: YES

User: Compare it with Rebase.
Answer: YES

User: According to the uploaded PDF, list Linux topics.
Answer: NO

Rules

- Answer ONLY YES or NO.
- No explanation.

Query

{query}

Answer
"""

        try:

            response = structured_rewrite_llm.invoke(prompt)

            answer = response.content.strip().lower()

            logger.info(
                "Rewrite Decision -> %s",
                response.rewrite,
            )

            return response.rewrite

        except Exception:

            logger.exception(
                "Rewrite decision failed."
            )

            return False