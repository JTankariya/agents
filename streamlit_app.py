"""Streamlit Community Cloud entrypoint with a generic failure boundary."""

from __future__ import annotations

import logging
import traceback
import uuid
from pathlib import Path


def _run() -> None:
    try:
        from apps.data_analyst.app import main

        main()
    except Exception as exc:  # Keep raw values out of both browser and logs.
        incident_id = uuid.uuid4().hex[:10]
        frames = traceback.extract_tb(exc.__traceback__)
        safe_stack = " > ".join(
            f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
            for frame in frames[-8:]
        )
        logging.error(
            "Unhandled app error; incident=%s type=%s stack=%s",
            incident_id,
            type(exc).__name__,
            safe_stack or "unavailable",
        )
        try:
            import streamlit as st

            st.error(
                "The application stopped safely because of an unexpected error. "
                f"Reference: `{incident_id}`"
            )
        except Exception:
            raise


if __name__ == "__main__":
    _run()
