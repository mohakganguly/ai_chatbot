"""
Conversation history utilities.
"""

from __future__ import annotations

from typing import List

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)


def build_history(
    messages: List,
    max_turns: int = 4,
) -> str:

    history = []

    recent = messages[:-1]

    recent = recent[-(max_turns * 2):]

    for message in recent:

        if isinstance(message, HumanMessage):

            history.append(
                f"User: {message.content}"
            )

        elif isinstance(message, AIMessage):

            history.append(
                f"Assistant: {message.content}"
            )

    return "\n".join(history)