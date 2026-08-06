"""
documents.py

Render attached documents.
"""

from __future__ import annotations

import streamlit as st


def render_documents(documents):
    """
    Returns
    -------
    str | None
        document_id requested for deletion
    """

    if not documents:
        return None

    st.markdown("##### 📎 Attached Documents")

    delete_document = None

    for document in documents:

        left, middle, right = st.columns([8, 2, 1])

        with left:

            st.markdown(
                f"📄 **{document.filename}**"
            )

        with middle:

            if document.status == "READY":

                st.success("Ready")

            elif document.status == "INDEXING":

                st.warning("Indexing")

            elif document.status == "FAILED":

                st.error("Failed")

            else:

                st.info(document.status)

        with right:

            if st.button(
                "✕",
                key=f"delete_{document.id}",
            ):

                delete_document = document.id

    return delete_document