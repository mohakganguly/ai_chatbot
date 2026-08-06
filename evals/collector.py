"""
collector.py

Collects chatbot outputs for evaluation.

This module is responsible for invoking the
production chatbot and converting its output
into a standardized EvaluationSample.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage
from evals.schemas import EvaluationSample
print("collector imported")

from graph.graph import chatbot

print("chatbot imported")


class Collector:
    """
    Executes the production chatbot.
    """

    def collect(
        self,
        question: str,
    ) -> EvaluationSample:

        start = perf_counter()

        result = chatbot.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=question
                    )
                ]
            }
        )

        latency = perf_counter() - start

        answer = result["messages"][-1].content

        retrieval = result.get(
            "retrieval_result"
        )

        contexts: list[str] = []

        citations = []

        if retrieval:

            contexts = [

                doc.page_content

                for doc in retrieval.documents

            ]

            citations = retrieval.citations

        metadata = {

            "planner_iterations":
                result.get("iteration", 0),

            "tool_calls":
                len(
                    result.get(
                        "tool_calls",
                        [],
                    )
                ),
        }

        return EvaluationSample(

            question=question,

            answer=answer,

            contexts=contexts,

            citations=citations,

            latency=latency,

            metadata=metadata,
        )