"""
routing.py

Query router responsible for deciding
whether a request should be handled by
general chat or by the agent.
"""

from __future__ import annotations

from core.llm import router_llm

from graph.routes import (
    GENERAL,
    AGENT,
)


ROUTER_PROMPT = """
You are a routing classifier.

Choose ONLY one route.

GENERAL
Use GENERAL for:

- Greetings
- Small talk
- Casual conversation
- Basic chit-chat
- Identity questions
- Thank you / Goodbye

Examples:

Hello
Hi
How are you?
Who are you?
Tell me a joke.
Good morning.

------------------------------------

AGENT

Use AGENT whenever the user requires
reasoning or tool usage.

Examples:

- Questions about uploaded files
- Summarization
- Retrieval
- Mathematical calculations
- Current information
- Web search
- Data analysis
- Python execution
- Comparisons
- Multi-step reasoning

Return ONLY

GENERAL

or

AGENT
"""


class QueryRouter:
    """
    Routes incoming user queries to either
    General Chat or the Agent workflow.
    """

    def __init__(self):

        self.llm = router_llm

    def route(
        self,
        query: str,
    ) -> str:

        response = self.llm.invoke(

            f"{ROUTER_PROMPT}\n\nUser Query:\n{query}"

        )

        decision = response.content.strip().upper()

        if decision == GENERAL.upper():
            return GENERAL

        return AGENT