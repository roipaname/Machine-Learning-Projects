"""
pages/platform_health.py — Backend / model health status dashboard.
"""
from __future__ import annotations

import streamlit as st

from components.html_components import empty_state, insight_card, section_title
from utils.api_client import get_health
from utils.formatters import format_timestamp


# ── Internal helpers ─────────────────────────────────────────

def _status_color(is_ok: bool) -> str:
    return "var(--color-accent-teal)" if is_ok else "var(--color-accent-red)"


# ── Public page entry point ──────────────────────────────────

def render() -> None:
    """Render the Platform Health page."""
    st.markdown(section_title("Platform Status", top_margin="0"), unsafe_allow_html=True)

    health = get_health()

    if not health:
        st.markdown(
            empty_state(
                icon="⚠",
                message="Backend unreachable.<br>Start the FastAPI server and refresh.",
            ),
            unsafe_allow_html=True,
        )
        return

    status     = health.get("status", "unknown")
    timestamp  = health.get("timestamp", "")
    version    = health.get("version", "—")
    models_ok  = health.get("models_loaded", False)

    api_color   = _status_color(status == "ok")
    model_color = _status_color(models_ok)
    model_label = "Loaded" if models_ok else "Not Ready"

    col_status, col_models, col_time = st.columns(3)

    with col_status:
        st.markdown(
            insight_card(
                label="API Status",
                value=f'<span style="color:{api_color}">{status.upper()}</span>',
                sub=f"v{version}",
            ),
            unsafe_allow_html=True,
        )

    with col_models:
        st.markdown(
            insight_card(
                label="AI Models",
                value=f'<span style="color:{model_color}">{model_label}</span>',
                sub="Advisor + Context Builder",
            ),
            unsafe_allow_html=True,
        )

    with col_time:
        st.markdown(
            insight_card(
                label="Last Check",
                value=f'<span style="font-size:16px">{format_timestamp(timestamp)}</span>',
                sub="UTC timestamp",
            ),
            unsafe_allow_html=True,
        )