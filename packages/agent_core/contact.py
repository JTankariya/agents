"""Formspree-backed contact and feedback submissions."""

from __future__ import annotations

import re
from typing import Any

import requests

from .errors import ContactError


_FORM_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{4,100}$")


def _endpoint(form_id: str) -> str:
    cleaned = form_id.strip()
    if not _FORM_ID_PATTERN.fullmatch(cleaned):
        raise ContactError("The Formspree form ID is not configured correctly.")
    return f"https://formspree.io/f/{cleaned}"


def submit_formspree(
    *,
    form_id: str,
    fields: dict[str, Any],
    timeout_seconds: float = 8.0,
) -> None:
    """Submit a small form and expose no provider response body to the user."""

    try:
        response = requests.post(
            _endpoint(form_id),
            data={key: str(value)[:5_000] for key, value in fields.items()},
            headers={
                "Accept": "application/json",
                "User-Agent": "agentic-data-analyst-playground/1.0",
            },
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise ContactError("The message service is temporarily unavailable.") from exc

    if response.status_code == 429:
        raise ContactError("The message service rate limit was reached. Try again later.")
    if response.status_code in {401, 403}:
        raise ContactError("The Formspree form is not authorized to accept submissions.")
    if response.status_code >= 400:
        raise ContactError("Formspree did not accept the submission.")
