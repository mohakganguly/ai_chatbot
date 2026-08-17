"""
collector.py

Collects chatbot outputs for evaluation.

This module invokes the production chatbot and converts
its output into a standardized EvaluationSample.
"""

from __future__ import annotations

import uuid
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage

from evals.schemas import EvaluationSample

from graph.graph import chatbot


class Collector:
    """
    Executes the production chatbot.
    """

    def collect(
        self,
        question: str,
        thread_id: str | None = None,
    ) -> EvaluationSample:

        # ======================================================
        # Evaluation Thread
        # ======================================================

        thread_id = (
            thread_id
            or str(uuid.uuid4())
        )

        # ======================================================
        # Invoke Production Chatbot
        # ======================================================

        start = perf_counter()

        result = chatbot.invoke(

            {
                "messages": [
                    HumanMessage(
                        content=question
                    )
                ]
            },

            config={
                "configurable": {
                    "thread_id": thread_id
                }
            },
        )

        latency = (
            perf_counter()
            - start
        )

        # ======================================================
        # Extract Final Answer
        # ======================================================

        messages = result.get(
            "messages",
            [],
        )

        if messages:

            answer = messages[-1].content

            if not isinstance(
                answer,
                str,
            ):
                answer = str(answer)

        else:

            answer = ""

        # ======================================================
        # Extract Retriever Tool Result
        # ======================================================

        retrieval = None

        contexts: list[str] = []

        citations: list[Any] = []

        tool_results = result.get(
            "tool_results",
            [],
        )

        for tool_result in tool_results:

            tool_name = getattr(
                tool_result,
                "tool_name",
                None,
            )

            if tool_name != "retriever":
                continue

            if not getattr(
                tool_result,
                "success",
                False,
            ):
                continue

            retrieval = getattr(
                tool_result,
                "output",
                None,
            )

            break

        # ======================================================
        # Extract Retrieval Data
        # ======================================================

        if retrieval is not None:

            documents = getattr(
                retrieval,
                "documents",
                [],
            ) or []

            contexts = [

                document.page_content

                for document in documents

                if getattr(
                    document,
                    "page_content",
                    None,
                )

            ]

            citations = getattr(
                retrieval,
                "citations",
                [],
            ) or []

        # ======================================================
        # Evaluation Metadata
        # ======================================================

        tool_history = result.get(
            "tool_history",
            [],
        )

        metadata = {

            "thread_id": thread_id,

            "planner_iterations": result.get(
                "iteration",
                0,
            ),

            "tool_calls": len(
                tool_history
            ),

            "retriever_used": (
                retrieval is not None
            ),

            "retrieved_documents": len(
                contexts
            ),

            "citation_count": len(
                citations
            ),

        }

        # ======================================================
        # Evaluation Sample
        # ======================================================

        return EvaluationSample(

            question=question,

            answer=answer,

            contexts=contexts,

            citations=citations,

            latency=latency,

            metadata=metadata,

        )