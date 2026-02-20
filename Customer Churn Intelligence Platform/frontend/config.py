"""
config.py — Application-wide constants and configuration.
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent
STYLES_DIR = ROOT_DIR / "styles"
CSS_FILE   = STYLES_DIR / "style.css"

# ── Backend ──────────────────────────────────────────────────
API_BASE        = "http://localhost:8000"
API_TIMEOUT_STD = 12    # seconds — context / health calls
API_TIMEOUT_LLM = 60    # seconds — LLM advice calls
API_TIMEOUT_BULK = 120  # seconds — bulk advice calls
BULK_MAX_IDS    = 20

# ── UI ───────────────────────────────────────────────────────
PAGE_TITLE  = "ZiyaBank · AI Advisor"
PAGE_ICON   = "🏦"
PAGES       = ["Account Advisor", "Bulk Risk View", "Platform Health"]
RISK_TIERS  = ["All", "Critical", "High", "Medium", "Low"]

# ── Risk badge CSS class mapping ─────────────────────────────
RISK_CLASS_MAP: dict[str, str] = {
    "Low":      "risk-badge--low",
    "Medium":   "risk-badge--medium",
    "High":     "risk-badge--high",
    "Critical": "risk-badge--critical",
}