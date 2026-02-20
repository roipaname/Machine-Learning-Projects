"""
components/html_components.py — Pure functions that return HTML strings.

Every function here is stateless and testable in isolation.
All CSS class names must match styles/ziyabank.css.
"""
from __future__ import annotations

from config import RISK_CLASS_MAP


# ── Primitives ───────────────────────────────────────────────

def risk_badge(tier: str) -> str:
    """Render a coloured risk-tier pill."""
    css_class = RISK_CLASS_MAP.get(tier, "risk-badge--medium")
    return f'<span class="risk-badge {css_class}">{tier}</span>'


def section_title(text: str, top_margin: str = "24px") -> str:
    """Small all-caps section label."""
    return (
        f'<div class="section-title" style="margin-top:{top_margin}">'
        f"{text}</div>"
    )


def thinking_state() -> str:
    """Animated loading indicator — no spinners per design spec."""
    return (
        '<div class="thinking-line"></div>'
        '<div class="thinking-text">Analyzing account signals…</div>'
    )


def empty_state(icon: str = "◎", message: str = "") -> str:
    """Centred empty-state placeholder."""
    return (
        '<div class="empty-state">'
        f'  <div class="empty-state__icon">{icon}</div>'
        f"  <p>{message}</p>"
        "</div>"
    )


# ── Health Bar ───────────────────────────────────────────────

def health_bar(health_pct: float) -> str:
    """
    Render a thin progress bar representing account health.

    Args:
        health_pct: 0–100 percentage value.
    """
    if health_pct >= 65:
        color = "var(--color-accent-teal)"
    elif health_pct >= 35:
        color = "var(--color-accent-amber)"
    else:
        color = "var(--color-accent-red)"

    return (
        '<div class="health-bar">'
        '  <div class="health-bar__meta">'
        '    <span class="health-bar__meta-label">Account Health</span>'
        f'   <span class="health-bar__meta-value" style="color:{color}">'
        f"      {health_pct:.0f}%"
        "    </span>"
        "  </div>"
        '  <div class="health-bar__track">'
        f'    <div class="health-bar__fill"'
        f'         style="width:{health_pct}%;background:{color}"></div>'
        "  </div>"
        "</div>"
    )


# ── Insight Card ─────────────────────────────────────────────

def insight_card(label: str, value: str, sub: str = "", extra_style: str = "") -> str:
    """
    Generic metric card.

    Args:
        label:       Small uppercase label above the value.
        value:       Large primary metric.
        sub:         Smaller descriptive text below the value.
        extra_style: Inline styles applied to the card root (e.g. text-align).
    """
    style_attr = f' style="{extra_style}"' if extra_style else ""
    sub_html   = f'<div class="card__sub">{sub}</div>' if sub else ""
    return (
        f'<div class="ziya-card"{style_attr}>'
        f'  <div class="card__label">{label}</div>'
        f'  <div class="card__value">{value}</div>'
        f"  {sub_html}"
        "</div>"
    )


# ── Account Row ──────────────────────────────────────────────

def account_row(company_name: str, industry: str, account_id: str) -> str:
    """Render a single account search result row."""
    return (
        '<div class="account-row">'
        f'  <div class="account-row__name">{company_name}</div>'
        f'  <div class="account-row__meta">'
        f"    {industry} &nbsp;·&nbsp; "
        f'    <code style="font-size:11px">{account_id}</code>'
        "  </div>"
        "</div>"
    )


# ── Signal Table ─────────────────────────────────────────────

def signal_table(signals: dict, label_map: dict[str, str] | None = None) -> str:
    """
    Render a key-value signal table.

    Args:
        signals:   Dict of raw signal key → value.
        label_map: Optional human-readable label overrides per key.
    """
    rows = ""
    for key, value in signals.items():
        label = (label_map or {}).get(key, key.replace("_", " ").title())
        rows += (
            '<div class="signal-table__row">'
            f'  <span class="signal-table__key">{label}</span>'
            f'  <span class="signal-table__value">{value}</span>'
            "</div>"
        )
    return f'<div class="signal-table">{rows}</div>'


# ── Driver Chips ─────────────────────────────────────────────

def driver_chips(drivers: list[str]) -> str:
    """Render churn driver tags."""
    chips = "".join(
        f'<span class="driver-chip">{driver}</span>' for driver in drivers
    )
    return f'<div style="padding:4px 0">{chips}</div>'


# ── AI Advisor Panel ─────────────────────────────────────────

def advisor_panel(advice_html: str) -> str:
    """
    Render the full AI advisor output panel.

    Args:
        advice_html: Pre-formatted inner HTML (ol/li or p).
    """
    return (
        '<div class="advisor-panel">'
        '  <div class="advisor-panel__header">'
        '    <div class="advisor-panel__dot"></div>'
        '    <span class="advisor-panel__title">Ziya AI Advisor</span>'
        '    <span class="advisor-panel__subtitle">Strategic Recommendations</span>'
        "  </div>"
        '  <div class="advisor-panel__body-wrap">'
        '    <div class="advisor-panel__body">'
        f"      {advice_html}"
        "    </div>"
        "  </div>"
        "</div>"
    )


# ── Brand Bar ────────────────────────────────────────────────

def brand_bar() -> str:
    """Sidebar brand / logo block."""
    return (
        '<div class="brand-bar">'
        '  <div class="brand-bar__dot"></div>'
        "  <div>"
        '    <div class="brand-bar__name">ZiyaBank</div>'
        '    <div class="brand-bar__sub">Intelligence Platform</div>'
        "  </div>"
        "</div>"
    )