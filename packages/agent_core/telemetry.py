"""Optional legacy analytics wrapper with privacy-safe scope."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def anonymous_shell_tracking(
    *,
    enabled: bool,
    password: str | None = None,
) -> Iterator[bool]:
    """Track only a static page shell, never uploader/chat/form values."""

    if not enabled:
        yield False
        return
    try:
        import streamlit_analytics

        kwargs: dict[str, Any] = {}
        if password:
            kwargs["unsafe_password"] = password
        streamlit_analytics.start_tracking(**kwargs)
    except Exception:
        yield False
        return
    try:
        yield True
    finally:
        try:
            streamlit_analytics.stop_tracking(**kwargs)
        except Exception:
            pass
