"""
utils/api_client.py — Thin wrapper around the ZiyaBank FastAPI backend.

All HTTP calls are centralised here so pages stay free of request logic.
Errors are surfaced as typed exceptions rather than raw requests exceptions,
giving callers a stable contract.
"""
from __future__ import annotations

import requests
import streamlit as st

from config import API_BASE, API_TIMEOUT_STD, API_TIMEOUT_LLM, API_TIMEOUT_BULK


class APIError(Exception):
    """Raised when the backend returns a non-2xx response."""
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class BackendUnavailableError(Exception):
    """Raised when the backend cannot be reached at all."""


# ── Internal helpers ─────────────────────────────────────────

def _get(path: str, timeout: int = API_TIMEOUT_STD) -> dict:
    """
    Perform a GET request against the backend.

    Raises:
        BackendUnavailableError: TCP / DNS failure.
        APIError: Non-2xx HTTP status.
    """
    try:
        response = requests.get(f"{API_BASE}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        raise BackendUnavailableError("Cannot reach backend.") from exc
    except requests.exceptions.HTTPError as exc:
        raise APIError(
            exc.response.text,
            status_code=exc.response.status_code,
        ) from exc


def _post(path: str, payload: dict, timeout: int = API_TIMEOUT_STD) -> dict:
    """
    Perform a POST request against the backend.

    Raises:
        BackendUnavailableError: TCP / DNS failure.
        APIError: Non-2xx HTTP status.
    """
    try:
        response = requests.post(
            f"{API_BASE}{path}", json=payload, timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        raise BackendUnavailableError("Cannot reach backend.") from exc
    except requests.exceptions.HTTPError as exc:
        raise APIError(
            exc.response.text,
            status_code=exc.response.status_code,
        ) from exc


# ── Public API ───────────────────────────────────────────────

def get_health() -> dict | None:
    """Fetch platform health. Returns None on any error (shown via st.error)."""
    try:
        return _get("/health")
    except BackendUnavailableError:
        st.error("⚠️  Cannot reach backend. Is the FastAPI server running?")
        return None
    except APIError as exc:
        st.error(f"Health check failed: {exc}")
        return None


def get_customer_context(customer_id: str) -> dict | None:
    """
    Fetch ML-enriched context for a customer.
    Returns None and surfaces an appropriate Streamlit message on failure.
    """
    try:
        return _get(f"/context/{customer_id}")
    except BackendUnavailableError:
        st.error("⚠️  Cannot reach backend. Is the FastAPI server running?")
        return None
    except APIError as exc:
        if exc.status_code == 404:
            st.warning(f"No data found for customer `{customer_id}`.")
        else:
            st.error(f"API error {exc.status_code}: {exc}")
        return None


def get_advice(customer_id: str) -> dict | None:
    """
    Run the full advisor pipeline (context → RAG → LLM) for one customer.
    Uses a longer timeout to accommodate LLM inference.
    """
    try:
        return _get(f"/advice/{customer_id.strip()}", timeout=API_TIMEOUT_LLM)
    except BackendUnavailableError:
        st.error("⚠️  Cannot reach backend. Is the FastAPI server running?")
        return None
    except APIError as exc:
        if exc.status_code == 404:
            st.warning(f"No data found for customer `{customer_id}`.")
        else:
            st.error(f"Advisor error {exc.status_code}: {exc}")
        return None


def get_bulk_advice(customer_ids: list[str]) -> dict | None:
    """Batch advisor for up to 20 customer IDs."""
    try:
        return _post(
            "/advise/bulk",
            {"customer_ids": customer_ids},
            timeout=API_TIMEOUT_BULK,
        )
    except BackendUnavailableError:
        st.error("⚠️  Cannot reach backend. Is the FastAPI server running?")
        return None
    except APIError as exc:
        st.error(f"Bulk advisor error {exc.status_code}: {exc}")
        return None


def get_accounts_by_company(company_name: str) -> dict | None:
    """Search accounts by company name."""
    try:
        return _get(f"/accounts/{company_name}")
    except BackendUnavailableError:
        st.error("⚠️  Cannot reach backend. Is the FastAPI server running?")
        return None
    except APIError as exc:
        if exc.status_code == 404:
            st.info("No accounts found for that company name.")
        else:
            st.error(f"Search error {exc.status_code}: {exc}")
        return None

def upload_document(
    filepath: str,
    original_name: str,
    doc_type: str = "other",
    segment: str = "All",
) -> dict | None:
    """Upload a local file to the RAG knowledge base."""
    try:
        with open(filepath, "rb") as f:
            response = requests.post(
                f"{API_BASE}/knowledge-base/upload",
                files={"file": (original_name, f)},
                data={"doc_type": doc_type, "segment": segment},
                timeout=API_TIMEOUT_STD,
            )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️  Cannot reach backend. Is the FastAPI server running?")
        return None
    except requests.exceptions.HTTPError as exc:
        st.error(f"Upload error {exc.response.status_code}: {exc.response.text}")
        return None