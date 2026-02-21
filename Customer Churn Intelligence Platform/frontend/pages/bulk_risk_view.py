"""
pages/bulk_risk_view.py — Batch churn analysis for multiple customers.
"""
from __future__ import annotations

import streamlit as st

from components.html_components import (
    insight_card,
    risk_badge,
    section_title,
    thinking_state,
)
from config import BULK_MAX_IDS, RISK_TIERS
from utils.api_client import get_bulk_advice
from utils.formatters import parse_bulk_ids, truncate


# ── Internal helpers ─────────────────────────────────────────

def _filter_and_sort(
    results: list[dict], risk_filter: str
) -> list[dict]:
    """Apply risk-tier filter then sort by descending churn probability."""
    if risk_filter != "All":
        results = [r for r in results if r.get("risk_tier") == risk_filter]
    return sorted(results, key=lambda r: r.get("churn_probability", 0), reverse=True)


def _render_result_card(result: dict) -> None:
    """Render an individual account result inside an expander."""
    customer_id = result["customer_id"]
    prob_pct    = result.get("churn_probability", 0) * 100
    tier        = result.get("risk_tier", "—")
    company_name = result.get("account_info", {}).get(
        "company_name", customer_id[:16] + "…"
    )
    drivers_text = ", ".join(result.get("top_churn_drivers", [])[:3]) or "—"
    advice_preview = truncate(str(result.get("advice", "—")))

    with st.expander(f"{company_name} — {prob_pct:.0f}% churn risk", expanded=False):
        col_metric, col_detail = st.columns([1, 3])

        with col_metric:
            badge = risk_badge(tier)
            st.markdown(
                insight_card(
                    label="Churn Risk",
                    value=f"{prob_pct:.0f}%",
                    extra_style="text-align:center;font-size:32px",
                )
                + f'<div style="margin-top:10px;text-align:center">{badge}</div>',
                unsafe_allow_html=True,
            )

        with col_detail:
            st.markdown(
                f"""
                <div class="ziya-card">
                  <div class="card__label">Top Drivers</div>
                  <div style="margin:6px 0;font-size:13px;color:var(--color-text-primary)">
                    {drivers_text}
                  </div>
                  <div class="card__label" style="margin-top:14px">AI Recommendation</div>
                  <div style="font-size:13px;line-height:1.7;
                              color:var(--color-text-primary);margin-top:4px">
                    {advice_preview}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_errors(errors: list[dict]) -> None:
    """Render per-customer errors at the bottom of the page."""
    if not errors:
        return
    st.markdown(section_title("Errors"), unsafe_allow_html=True)
    for err in errors:
        st.markdown(
            f'<div style="font-size:12px;color:var(--color-accent-red);margin-bottom:4px">'
            f"⚠ {err['customer_id']}: {err['error']}</div>",
            unsafe_allow_html=True,
        )


# ── Public page entry point ──────────────────────────────────

def render() -> None:
    """Render the Bulk Risk View page."""
    st.markdown(section_title("Bulk Risk Analysis", top_margin="0"), unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:13px;color:var(--color-text-muted);'
        f'margin-bottom:18px;line-height:1.6">'
        f"Enter up to {BULK_MAX_IDS} customer IDs (one per line) "
        f"to run batch churn analysis.</div>",
        unsafe_allow_html=True,
    )

    raw_ids = st.text_area(
        "Customer IDs",
        placeholder="customer-id-1\ncustomer-id-2\ncustomer-id-3\n…",
        height=180,
        label_visibility="collapsed",
    )

    risk_filter = st.selectbox("Filter by risk tier", RISK_TIERS)

    st.markdown('<div class="btn--primary">', unsafe_allow_html=True)
    run_bulk = st.button("Run Bulk Analysis")
    st.markdown("</div>", unsafe_allow_html=True)

    if not run_bulk:
        return

    if not raw_ids.strip():
        st.warning("Please enter at least one customer ID.")
        return

    customer_ids = parse_bulk_ids(raw_ids, max_count=BULK_MAX_IDS)

    with st.spinner(""):
        st.markdown(thinking_state(), unsafe_allow_html=True)
        bulk_data = get_bulk_advice(customer_ids)

    if not bulk_data:
        return  # error surfaced by api_client

    results = _filter_and_sort(bulk_data.get("results", []), risk_filter)
    errors  = bulk_data.get("errors", [])

    st.markdown(
        section_title(f"{len(results)} accounts &nbsp;·&nbsp; {len(errors)} errors"),
        unsafe_allow_html=True,
    )

    for result in results:
        _render_result_card(result)

    _render_errors(errors)