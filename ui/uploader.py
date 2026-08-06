"""
uploader.py
"""

from __future__ import annotations

import streamlit as st


def render_uploader():

    st.markdown("---")

    st.markdown("##### Add documents to this conversation")

    uploaded_files = st.file_uploader(

        "Attach documents",

        type=[
            "pdf",
            "docx",
            "txt",
            "md",
            "html",
        ],

        accept_multiple_files=True,

        label_visibility="collapsed",
    )

    return uploaded_files