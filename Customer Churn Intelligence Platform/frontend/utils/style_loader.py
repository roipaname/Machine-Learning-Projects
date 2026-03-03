"""
utils/style_loader.py — Load and inject the external CSS file into Streamlit.
"""
from __future__ import annotations

import streamlit as st

from config import CSS_FILE


def inject_css() -> None:
    """
    Read the ZiyaBank stylesheet from disk and inject it into the Streamlit page.

    Raises a Streamlit error widget (rather than crashing) if the file is missing,
    so the app stays functional even if styles fail to load.
    """
    if not CSS_FILE.exists():
        st.error(
            f"Stylesheet not found: `{CSS_FILE}`. "
            "Ensure `styles/ziyabank.css` is present in the project directory."
        )
        return

    css_content = CSS_FILE.read_text(encoding="utf-8")
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)