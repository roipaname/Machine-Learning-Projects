"""
utils/formatters.py — Pure text / data transformation helpers.

No Streamlit or HTTP dependencies — safe to unit-test independently.
"""
from __future__ import annotations


def format_advice_as_html(advice_text: str) -> str:
    """
    Convert numbered plain-text advice into a semantic HTML ordered list.

    Handles both:
    - "1. Do something"  (leading digit + period)
    - Plain sentences    (wrapped in a <p> fallback)
    """
    lines = [line.strip() for line in advice_text.strip().splitlines() if line.strip()]
    items = [_strip_leading_number(line) for line in lines if _strip_leading_number(line)]

    if not items:
        return f"<p>{advice_text}</p>"

    li_tags = "".join(f"<li>{item}</li>" for item in items)
    return f"<ol>{li_tags}</ol>"


def _strip_leading_number(line: str) -> str:
    """Remove a leading 'N.' or 'N) ' prefix from a line."""
    stripped = line.lstrip("0123456789").lstrip(". )").strip()
    return stripped


def format_churn_probability(probability: float) -> str:
    """Convert a 0–1 float to a display percentage string."""
    return f"{probability * 100:.0f}%"


def format_payment_rate(rate: float | None) -> str:
    """Format payment rate, returning em-dash for missing data."""
    if not isinstance(rate, (int, float)):
        return "—"
    return f"{rate * 100:.0f}%"


def format_timestamp(iso_timestamp: str) -> str:
    """Strip the T separator from ISO timestamps for display."""
    return iso_timestamp[:19].replace("T", " ") if iso_timestamp else "—"


def parse_bulk_ids(raw_text: str, max_count: int) -> list[str]:
    """
    Parse a newline-separated string of customer IDs.

    Args:
        raw_text:  Raw textarea content.
        max_count: Maximum number of IDs to return.

    Returns:
        Deduplicated, trimmed list of IDs, capped at max_count.
    """
    ids = [line.strip() for line in raw_text.splitlines() if line.strip()]
    seen: set[str] = set()
    unique: list[str] = []
    for cid in ids:
        if cid not in seen:
            seen.add(cid)
            unique.append(cid)
    return unique[:max_count]


def truncate(text: str, max_chars: int = 400, ellipsis: str = "…") -> str:
    """Truncate long strings for preview display."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + ellipsis


def build_export_text(customer_id: str, advice: str) -> str:
    """Format advisory text suitable for file export."""
    separator = "=" * 40
    return f"ZiyaBank AI Advisory\n{separator}\nCustomer: {customer_id}\n\n{advice}"