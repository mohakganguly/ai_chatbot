"""
messages.py

Responsible for rendering the chat conversation.
"""

from __future__ import annotations

import streamlit as st


def render_messages(
    messages,
):
    """
    Render every chat message.
    """

    for message in messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )