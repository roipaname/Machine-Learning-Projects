"""
pages/knowledge_base.py — Upload documents into the RAG vector store.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from components.html_components import section_title
from utils.api_client import upload_document


def render() -> None:
    st.markdown(section_title("Knowledge Base", top_margin="0px"), unsafe_allow_html=True)
    st.markdown(
        '<p style="color:var(--color-text-muted);margin-bottom:24px">'
        "Upload strategy documents, playbooks, or reports to enrich Ziya AI's retrieval context.</p>",
        unsafe_allow_html=True,
    )

    # ── Upload widget ────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
        help="Supported: .txt · .md · .pdf · .docx",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        doc_type = st.selectbox(
            "Document type",
            ["playbook", "analyst_report", "case_study", "policy", "other"],
        )
    with col2:
        segment = st.selectbox(
            "Target segment",
            ["All", "Enterprise", "Mid-Market", "SMB"],
        )

    if not uploaded_files:
        st.info("No files selected yet.")
        return

    if st.button("Upload to Knowledge Base", type="primary"):
        results = []
        progress = st.progress(0, text="Uploading…")

        for i, uf in enumerate(uploaded_files):
            # Write to a temp file so the backend can read it by path,
            # or send raw bytes — here we POST multipart via api_client.
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uf.name).suffix
            ) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name

            result = upload_document(
                filepath=tmp_path,
                original_name=uf.name,
                doc_type=doc_type,
                segment=segment,
            )
            results.append((uf.name, result))
            progress.progress(
                (i + 1) / len(uploaded_files),
                text=f"Processed {i + 1}/{len(uploaded_files)}",
            )

        progress.empty()
        _render_results(results)


def _render_results(results: list[tuple[str, dict | None]]) -> None:
    st.markdown("---")
    for filename, result in results:
        if result is None:
            st.error(f"❌  **{filename}** — upload failed (check backend logs)")
        else:
            doc_id = result.get("doc_id", "—")
            chunks = result.get("chunks_added", "—")
            st.success(f"✅  **{filename}** ingested as `{doc_id}` · {chunks} chunk(s) added")