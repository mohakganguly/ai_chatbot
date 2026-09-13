"""
sidebar.py

Sidebar UI for the Enterprise AI Assistant.

Responsible for:
- New Chat button
- Conversation history
- Chat selection
"""

from __future__ import annotations

import streamlit as st


def render_sidebar(conversations):
    """
    Render sidebar.

    Returns:
        selected_thread_id | None
    """

    st.sidebar.title("Enterprise AI Assistant")
    st.sidebar.write(
        f"Thread ID: {st.session_state['thread_id']}"
    )
    st.sidebar.divider()

    new_chat = st.sidebar.button(
        "➕ New Chat",
        use_container_width=True,
    )

    st.sidebar.divider()

    st.sidebar.subheader("Recent Chats")

    selected_thread = None

    for conversation in conversations:

        if st.sidebar.button(
            conversation.title,
            key=f"chat_{conversation.thread_id}",
            use_container_width=True,
        ):
            selected_thread = conversation.thread_id

    return new_chat, selected_thread