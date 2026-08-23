"""
state.py

Shared state for the Enterprise AI Assistant.
"""

from __future__ import annotations

from typing import Annotated, TypedDict, Literal,Any

from langgraph.graph.message import add_messages

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from agents.schemas import ToolCall
from tools.base import ToolResult

class Citation(TypedDict):
    source: str
    page: int | str
    retrieval_score: float
    reranker_score: float


class ChatState(TypedDict):

    # Chat history
    messages: Annotated[list, add_messages]

    # Original user query
    query: str

    # Router decision
    route: Literal["general", "rag"]

    # Query after rewriting
    retrieval_query: str

    # Retrieved documents
    retrieved_documents: list[Document]

    # Final prompt sent to the LLM
    prompt: str

    # Source citations
    citations: list[Citation]


from agents.schemas import ObservationDecision
from typing import NotRequired

class AgentState(TypedDict):
    """
    Shared state for the agent graph.

    Every node receives this state,
    updates the fields it owns,
    and returns the modified state.
    """

    # Original user request
    query: str

    # Router decision
    # route: Literal[
    #     "general",
    #     "agent",
    # ]
    # Chat history
    messages: Annotated[list[BaseMessage], add_messages]

    # Latest planner decision
    tool_call: ToolCall | None

    # Latest tool result
    tool_result: ToolResult | None

    # Final response returned to the user
    final_answer: str | None

    # History of tool calls
    tool_history: list[ToolCall]

    # History of tool results
    tool_results: list[ToolResult]

    tool_guardrail_status: NotRequired[str]
    tool_guardrail_reason: NotRequired[str | None]

    observation: ObservationDecision | None

    # Number of reasoning iterations
    iteration: int

    # Current graph status
    status: Literal[
        "PLANNING",
        "EXECUTING",
        "OBSERVING",
        "ANSWERING",
        "FINISHED",
    ]

    # Error message (if any)
    error: str | None


    # ==========================================================
    # Guardrails
    # ==========================================================

    # Result of input safety validation
    guardrail_status: Literal[
        "SAFE",
        "BLOCKED",
    ] | None

    # Reason the request was blocked
    guardrail_reason: str | None

    # Extra metadata for future use
    metadata: dict[str, Any]

