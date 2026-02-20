"""
app.py — ZiyaBank Intelligence Platform entry point.

Run with:
    streamlit run app.py

Project layout:
    app.py                  ← this file (Streamlit entry point)
    config.py               ← constants & paths
    styles/
        ziyabank.css        ← all CSS (design system)
    components/
        html_components.py  ← pure HTML-string builders
    utils/
        api_client.py       ← HTTP calls to FastAPI backend
        formatters.py       ← pure data/text transformation helpers
        style_loader.py     ← CSS injection into Streamlit
    pages/
        account_advisor.py  ← single-account analysis + AI advisor
        bulk_risk_view.py   ← batch risk analysis
        platform_health.py  ← backend health dashboard
"""

import streamlit as st

from config import PAGE_ICON, PAGE_TITLE, PAGES
from components.html_components import brand_bar
from pages import account_advisor, bulk_risk_view, platform_health
from utils.style_loader import inject_css

# ── Page config (must be first Streamlit call) ───────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject external CSS ───────────────────────────────────────
inject_css()

# ── Sidebar navigation ────────────────────────────────────────
with st.sidebar:
    st.markdown(brand_bar(), unsafe_allow_html=True)

    page = st.radio(
        "",
        PAGES,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:11px;color:var(--color-text-muted);padding-top:4px">'
        "Ziya AI · v1.0.0</div>",
        unsafe_allow_html=True,
    )

# ── Page routing ──────────────────────────────────────────────
_PAGE_MAP = {
    "Account Advisor":  account_advisor.render,
    "Bulk Risk View":   bulk_risk_view.render,
    "Platform Health":  platform_health.render,
}

render_page = _PAGE_MAP.get(page)
if render_page:
    render_page()