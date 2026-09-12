"""Central configuration for the playground."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Runtime limits chosen for a small free-tier Streamlit deployment."""

    model: str = "openai/gpt-oss-20b"
    max_upload_bytes: int = 5 * 1024 * 1024
    max_rows: int = 50_000
    max_columns: int = 100
    max_column_name_chars: int = 128
    max_total_column_name_chars: int = 6_000
    max_dataframe_bytes: int = 96 * 1024 * 1024
    max_runs_per_session: int = 3
    max_prompt_chars: int = 1_200
    llm_timeout_seconds: float = 30.0
    llm_max_completion_tokens: int = 1_200
    sandbox_timeout_seconds: float = 10.0
    sandbox_extra_memory_bytes: int = 512 * 1024 * 1024
    sandbox_max_code_chars: int = 5_000
    sandbox_max_ast_nodes: int = 600
    global_llm_requests_per_minute: int = 20

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "AppSettings":
        """Create settings from secrets while retaining conservative defaults."""

        values = values or {}
        defaults = cls()

        def positive_int(name: str, default: int) -> int:
            raw = values.get(name, default)
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                return default
            return parsed if parsed > 0 else default

        model = str(values.get("GROQ_MODEL", defaults.model)).strip() or defaults.model
        return cls(
            model=model,
            max_runs_per_session=min(
                positive_int("MAX_RUNS_PER_SESSION", defaults.max_runs_per_session),
                10,
            ),
            global_llm_requests_per_minute=min(
                positive_int(
                    "GLOBAL_LLM_REQUESTS_PER_MINUTE",
                    defaults.global_llm_requests_per_minute,
                ),
                60,
            ),
        )
