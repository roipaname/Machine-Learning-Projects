"""
pages/account_advisor.py — Single-account churn analysis and AI recommendations.
"""
from __future__ import annotations

import time

import streamlit as st

from components.html_components import (
    account_row,
    advisor_panel,
    driver_chips,
    health_bar,
    insight_card,
    risk_badge,
    section_title,
    signal_table,
    thinking_state,
    empty_state,
)
from utils.api_client import get_accounts_by_company, get_advice, get_customer_context
from utils.formatters import (
    build_export_text,
    format_advice_as_html,
    format_churn_probability,
    format_payment_rate,
)


# ── Internal helpers ─────────────────────────────────────────

def _render_company_search() -> None:
    """Collapsible helper that lets users look up customer IDs by company name."""
    with st.expander("🔍  Find accounts by company name", expanded=False):
        company = st.text_input(
            "Company name",
            placeholder="e.g. Acme Holdings",
            key="company_input",
        )
        if not company:
            return

        data = get_accounts_by_company(company)
        if not data:
            return

        accounts = data.get("accounts", [])
        if not accounts:
            st.markdown(
                '<div class="thinking-text">No accounts found.</div>',
                unsafe_allow_html=True,
            )
            return

        for acc in accounts[:8]:
            st.markdown(
                account_row(
                    company_name=acc.get("company_name", "Unknown"),
                    industry=acc.get("industry", "—"),
                    account_id=str(acc.get("account_id", "")),
                ),
                unsafe_allow_html=True,
            )


def _render_account_header(
    company_name: str,
    industry: str,
    tier: str,
    risk_tier: str,
    churn_pct: float,
) -> None:
    """Display the account name, metadata row, and health bar."""
    badge = risk_badge(risk_tier)
    st.markdown(
        f"""
        <div style="margin: 18px 0 6px;">
          <div style="font-size:22px;font-weight:600;color:var(--color-text-primary)">
            {company_name}
          </div>
          <div style="font-size:13px;color:var(--color-text-muted);margin-top:4px">
            {industry} &nbsp;·&nbsp; {tier} &nbsp;·&nbsp; {badge}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(health_bar(100 - churn_pct), unsafe_allow_html=True)


def _render_signal_cards(
    engagement: dict,
    support: dict,
    billing: dict,
) -> None:
    """Render the three insight metric cards side-by-side."""
    col_eng, col_sup, col_bil = st.columns(3)

    with col_eng:
        events = engagement.get("total_events", "—")
        logins = engagement.get("login_count", "—")
        st.markdown(
            insight_card("Engagement", str(events), f"Total events &nbsp;·&nbsp; {logins} logins"),
            unsafe_allow_html=True,
        )

    with col_sup:
        tickets  = support.get("total_tickets", "—")
        avg_sat  = support.get("avg_satisfaction_score", 0)
        sat_fmt  = f"{avg_sat:.1f}" if isinstance(avg_sat, (int, float)) else "—"
        high_pri = support.get("high_priority_tickets", 0)
        st.markdown(
            insight_card(
                "Support",
                str(tickets),
                f"Tickets &nbsp;·&nbsp; {high_pri} high-priority &nbsp;·&nbsp; CSAT {sat_fmt}",
            ),
            unsafe_allow_html=True,
        )

    with col_bil:
        paid_fmt = format_payment_rate(billing.get("payment_rate"))
        late_avg = billing.get("avg_days_late", 0)
        st.markdown(
            insight_card(
                "Billing",
                paid_fmt,
                f"Payment rate &nbsp;·&nbsp; avg {late_avg:.0f}d late",
            ),
            unsafe_allow_html=True,
        )


def _render_advisor_actions(customer_id: str, advice_text: str) -> None:
    """Render the three post-advice action buttons."""
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.button("Create Action Plan", use_container_width=True)

    with col_b:
        st.button("Send to Relationship Manager", use_container_width=True)

    with col_c:
        st.download_button(
            "Export Advisory (.txt)",
            data=build_export_text(customer_id, advice_text),
            file_name=f"advisory_{customer_id[:8]}.txt",
            mime="text/plain",
            use_container_width=True,
        )


def _render_signal_breakdown(
    engagement: dict, support: dict, billing: dict
) -> None:
    """Collapsible detailed signal tables."""
    with st.expander("Detailed Signal Breakdown", expanded=False):
        if engagement:
            st.markdown(section_title("Engagement", top_margin="8px"), unsafe_allow_html=True)
            st.markdown(signal_table(engagement), unsafe_allow_html=True)

        if support:
            st.markdown(section_title("Support"), unsafe_allow_html=True)
            st.markdown(signal_table(support), unsafe_allow_html=True)

        if billing:
            st.markdown(section_title("Billing"), unsafe_allow_html=True)
            st.markdown(signal_table(billing), unsafe_allow_html=True)


# ── Public page entry point ──────────────────────────────────

def render() -> None:
    """Render the Account Advisor page."""

    # ── Search bar ───────────────────────────────────────────
    st.markdown(section_title("Account Search", top_margin="0"), unsafe_allow_html=True)

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        customer_id = st.text_input(
            "Customer ID",
            placeholder="Enter customer ID or search…",
            label_visibility="collapsed",
            key="cid_input",
        )
    with col_btn:
        st.markdown('<div class="btn--primary">', unsafe_allow_html=True)
        run = st.button("Analyse", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    _render_company_search()

    # ── Guard: no input yet ───────────────────────────────────
    if not customer_id and not run:
        st.markdown(
            empty_state(
                icon="◎",
                message="Select an account to view<br>strategic insights and AI recommendations.",
            ),
            unsafe_allow_html=True,
        )
        return

    if not customer_id:
        st.warning("Please enter a customer ID.")
        return

    # ── Fetch context ────────────────────────────────────────
    with st.spinner(""):
        st.markdown(thinking_state(), unsafe_allow_html=True)
        ctx = get_customer_context(customer_id)

    if not ctx:
        return  # error already surfaced by api_client

    # ── Unpack context ───────────────────────────────────────
    account_info = ctx.get("account_info", {})
    engagement   = ctx.get("engagement_signals", {})
    support      = ctx.get("support_signals", {})
    billing      = ctx.get("billing_signals", {})
    drivers      = ctx.get("top_churn_drivers", [])

    company_name = account_info.get("company_name", customer_id)
    industry     = account_info.get("industry", "—")
    tier         = account_info.get("account_tier", "—").title()
    churn_prob   = ctx.get("churn_probability", 0)
    risk_tier    = ctx.get("risk_tier", "Unknown")
    churn_pct    = churn_prob * 100

    # ── Account header ────────────────────────────────────────
    _render_account_header(company_name, industry, tier, risk_tier, churn_pct)

    # ── Signal cards ─────────────────────────────────────────
    st.markdown(section_title("Signals"), unsafe_allow_html=True)
    _render_signal_cards(engagement, support, billing)

    # ── Churn drivers ─────────────────────────────────────────
    if drivers:
        st.markdown(section_title("Top Churn Drivers"), unsafe_allow_html=True)
        st.markdown(driver_chips(drivers), unsafe_allow_html=True)

    # ── AI Advisor ─────────────────────────────────────────────
    st.markdown(section_title("AI Advisor"), unsafe_allow_html=True)

    if st.button("Generate Strategic Recommendations"):
        with st.spinner(""):
            st.markdown(thinking_state(), unsafe_allow_html=True)
            time.sleep(0.3)  # brief UX pause so the thinking state registers
            advice_data = get_advice(customer_id)

        if advice_data:
            st.session_state["last_advice"] = advice_data.get(
                "advice", "No advice returned."
            )

    if "last_advice" in st.session_state:
        advice_text = st.session_state["last_advice"]
        st.markdown(
            advisor_panel(format_advice_as_html(advice_text)),
            unsafe_allow_html=True,
        )
        st.markdown('<div style="margin-top:16px">', unsafe_allow_html=True)
        _render_advisor_actions(customer_id, advice_text)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Detailed signal breakdown ─────────────────────────────
    _render_signal_breakdown(engagement, support, billing)