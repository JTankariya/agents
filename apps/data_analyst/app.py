"""Streamlit UI for the Groq-powered data analyst agent."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import pandas as pd
import streamlit as st

from packages.agent_core.agent import DataAnalystAgent
from packages.agent_core.config import AppSettings
from packages.agent_core.contact import submit_formspree
from packages.agent_core.contracts import AgentRunResult, TraceEvent
from packages.agent_core.data_profile import CsvLoadResult, load_csv_bytes
from packages.agent_core.errors import ContactError, DatasetError, GatewayError
from packages.agent_core.groq_gateway import GroqGateway
from packages.agent_core.rate_limit import SlidingWindowRateLimiter
from packages.agent_core.rendering import render_output
from packages.agent_core.sandbox import SandboxPolicy
from packages.agent_core.telemetry import anonymous_shell_tracking


def _secrets_dict() -> dict[str, Any]:
    try:
        return {key: st.secrets[key] for key in st.secrets}
    except Exception:
        return {}


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    return default


@st.cache_resource(show_spinner=False)
def _shared_limiter(max_requests: int) -> SlidingWindowRateLimiter:
    return SlidingWindowRateLimiter(max_requests=max_requests, window_seconds=60.0)


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "session_id": uuid.uuid4().hex,
        "runs_used": 0,
        "file_hash": None,
        "dataframe": None,
        "load_result": None,
        "last_run": None,
        "feedback_sent": set(),
        "contact_sent": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_header(legacy_analytics_enabled: bool, analytics_password: str) -> None:
    with anonymous_shell_tracking(
        enabled=legacy_analytics_enabled,
        password=analytics_password or None,
    ):
        st.title("Agentic Data Analyst Playground")
        st.caption(
            "Upload a CSV, ask a data question, inspect the generated pandas code, "
            "and run it in an isolated local sandbox."
        )


def _load_uploaded_file(uploaded_file: Any, settings: AppSettings) -> None:
    file_bytes = uploaded_file.getvalue()
    digest = hashlib.sha256(file_bytes).hexdigest()
    if digest == st.session_state.file_hash:
        return
    try:
        load_result = load_csv_bytes(file_bytes, settings)
    except DatasetError:
        st.session_state.file_hash = None
        st.session_state.dataframe = None
        st.session_state.load_result = None
        st.session_state.last_run = None
        raise

    st.session_state.file_hash = digest
    st.session_state.dataframe = load_result.dataframe
    st.session_state.load_result = load_result
    st.session_state.last_run = None
    st.session_state.feedback_sent = set()


def _trace_renderer(event: TraceEvent) -> None:
    icon = {
        "step": "➡️",
        "code": "🧩",
        "warning": "⚠️",
        "error": "❌",
        "success": "✅",
    }.get(event.kind, "•")
    st.write(f"{icon} **{event.label}** — {event.detail}")
    if event.code:
        st.code(event.code, language="python", line_numbers=True)


def _build_agent(secrets: dict[str, Any], settings: AppSettings) -> DataAnalystAgent:
    api_key = str(secrets.get("GROQ_API_KEY", "")).strip()
    if not api_key:
        raise GatewayError(
            "GROQ_API_KEY is missing. Add it in Streamlit Community Cloud secrets."
        )
    gateway = GroqGateway(
        api_key=api_key,
        model=settings.model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_completion_tokens=settings.llm_max_completion_tokens,
        limiter=_shared_limiter(settings.global_llm_requests_per_minute),
    )
    policy = SandboxPolicy(
        timeout_seconds=settings.sandbox_timeout_seconds,
        extra_memory_bytes=settings.sandbox_extra_memory_bytes,
        max_code_chars=settings.sandbox_max_code_chars,
        max_ast_nodes=settings.sandbox_max_ast_nodes,
        max_result_bytes=settings.max_dataframe_bytes,
        max_rows=settings.max_rows,
        max_columns=settings.max_columns,
    )
    return DataAnalystAgent(gateway=gateway, sandbox_policy=policy)


def _render_dataset(load_result: CsvLoadResult) -> None:
    df = load_result.dataframe
    left, middle, right = st.columns(3)
    left.metric("Rows loaded", f"{len(df):,}")
    middle.metric("Columns", f"{df.shape[1]:,}")
    right.metric("Memory", f"{load_result.memory_bytes / (1024 * 1024):.1f} MB")

    if load_result.truncated:
        st.warning(
            "The CSV contains more than 50,000 rows. Only the first 50,000 were loaded."
        )
    with st.expander("Local dataset preview", expanded=False):
        st.caption("This preview remains in the app process and is not sent to Groq.")
        st.dataframe(df.head(50), width="stretch", height=320)
        st.code(
            "\n".join(f"{column}: {dtype}" for column, dtype in df.dtypes.items()),
            language="text",
        )


def _render_last_result(
    run_result: AgentRunResult,
    *,
    formspree_form_id: str,
    session_id: str,
) -> None:
    st.subheader("Agent result")
    if run_result.status == "guardrail":
        st.warning(run_result.message)
        return
    if run_result.status == "failed":
        st.error(run_result.message)
        if run_result.errors:
            with st.expander("Sanitized diagnostics"):
                for error in run_result.errors:
                    st.code(error, language="text")
        return

    if run_result.output is not None:
        render_output(run_result.output)
    if run_result.plan is not None:
        with st.expander("Generated code", expanded=False):
            st.code(run_result.plan.code, language="python", line_numbers=True)

    feedback_key = f"feedback_{st.session_state.file_hash}_{st.session_state.runs_used}"
    st.markdown("**Rate this analysis**")
    selected = st.feedback("stars", key=feedback_key)
    if selected is not None and feedback_key not in st.session_state.feedback_sent:
        st.session_state.feedback_sent.add(feedback_key)
        if formspree_form_id:
            try:
                submit_formspree(
                    form_id=formspree_form_id,
                    fields={
                        "submission_type": "agent_feedback",
                        "rating": selected + 1,
                        "session_hash": hashlib.sha256(session_id.encode()).hexdigest()[:12],
                        "agent_status": run_result.status,
                        "attempts": run_result.attempts,
                    },
                )
                st.caption("Thank you — the rating was recorded.")
            except ContactError:
                st.caption(
                    "The rating is visible for this session but could not be persisted."
                )
        else:
            st.caption("Add FORMSPREE_FORM_ID to persist ratings outside this session.")


def _render_contact_form(formspree_form_id: str) -> None:
    st.divider()
    st.subheader("Contact the creator")
    if not formspree_form_id:
        st.info("Configure FORMSPREE_FORM_ID in Streamlit secrets to enable this form.")
        return
    if st.session_state.contact_sent:
        st.success("A contact message has already been submitted from this browser session.")
        return

    st.caption("Submitting this form sends the entered contact details to Formspree.")
    with st.form("contact_form", clear_on_submit=True):
        name = st.text_input("Name", max_chars=100)
        email = st.text_input("Email", max_chars=254)
        message = st.text_area("Message", max_chars=2_000, height=140)
        honeypot = st.text_input(
            "Website",
            key="contact_website",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send message", width="stretch")

    if submitted:
        if honeypot:
            st.session_state.contact_sent = True
            st.success("Message sent.")
            return
        if not name.strip() or "@" not in email or not message.strip():
            st.warning("Enter a name, a valid email address, and a message.")
            return
        try:
            submit_formspree(
                form_id=formspree_form_id,
                fields={
                    "submission_type": "contact",
                    "name": name.strip(),
                    "email": email.strip(),
                    "message": message.strip(),
                },
            )
            st.session_state.contact_sent = True
            st.success("Message sent successfully.")
        except ContactError as exc:
            st.error(str(exc))


def main() -> None:
    st.set_page_config(
        page_title="Agentic Data Analyst Playground",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session_state()
    secrets = _secrets_dict()
    settings = AppSettings.from_mapping(secrets)

    legacy_analytics_enabled = _as_bool(secrets.get("ENABLE_LEGACY_ANALYTICS", False))
    analytics_password = str(secrets.get("ANALYTICS_PASSWORD", ""))
    formspree_form_id = str(secrets.get("FORMSPREE_FORM_ID", "")).strip()

    _render_header(legacy_analytics_enabled, analytics_password)

    with st.sidebar:
        st.header("Demo limits")
        remaining = max(0, settings.max_runs_per_session - st.session_state.runs_used)
        st.metric("Agent runs remaining", remaining)
        st.caption(f"Model: `{settings.model}`")
        st.caption("One failed execution may use one automatic correction call.")
        st.divider()
        st.markdown(
            "**Privacy contract**\n\n"
            "Groq receives your written request plus column names, dtypes, "
            "dimensions, and up to five rows of type placeholders. Original CSV "
            "cell values are not automatically included. Do not paste sensitive "
            "values into the request."
        )
        st.markdown(
            "**Try asking**\n\n"
            "- Summarize missing values by column\n"
            "- Plot a correlation matrix for numeric columns\n"
            "- Group revenue by region and sort descending\n"
            "- Clean duplicate rows and return an updated dataset"
        )

    uploaded_file = st.file_uploader(
        "Upload a CSV (maximum 5 MB)",
        type=["csv"],
        accept_multiple_files=False,
        max_upload_size=5,
    )

    if uploaded_file is None:
        st.info("Upload a CSV to activate the analyst.")
        _render_contact_form(formspree_form_id)
        return

    try:
        _load_uploaded_file(uploaded_file, settings)
    except DatasetError as exc:
        st.error(str(exc))
        _render_contact_form(formspree_form_id)
        return

    load_result: CsvLoadResult = st.session_state.load_result
    dataframe: pd.DataFrame = st.session_state.dataframe
    _render_dataset(load_result)

    prompt = st.chat_input(
        "Ask a question about the uploaded dataset",
        max_chars=settings.max_prompt_chars,
        disabled=st.session_state.runs_used >= settings.max_runs_per_session,
    )

    if prompt:
        if st.session_state.runs_used >= settings.max_runs_per_session:
            st.warning("This browser session has used all available demo runs.")
        elif not prompt.strip():
            st.warning("Enter a data-analysis request.")
        else:
            try:
                agent = _build_agent(secrets, settings)
            except GatewayError as exc:
                st.error(str(exc))
            else:
                st.session_state.runs_used += 1
                with st.status("Running the data analyst agent…", expanded=True) as status:
                    run_result = agent.run(
                        user_request=prompt.strip(),
                        dataframe=dataframe,
                        on_trace=_trace_renderer,
                    )
                    if run_result.status == "success":
                        status.update(
                            label="Analysis complete",
                            state="complete",
                            expanded=False,
                        )
                    elif run_result.status == "guardrail":
                        status.update(
                            label="Request declined by scope guardrail",
                            state="complete",
                            expanded=True,
                        )
                    else:
                        status.update(
                            label="Analysis stopped safely",
                            state="error",
                            expanded=True,
                        )
                st.session_state.last_run = run_result

    if st.session_state.last_run is not None:
        _render_last_result(
            st.session_state.last_run,
            formspree_form_id=formspree_form_id,
            session_id=st.session_state.session_id,
        )

    if st.session_state.runs_used >= settings.max_runs_per_session:
        st.info(
            "The three-run session limit has been reached. This is a portfolio-demo "
            "guardrail, not an identity-based quota."
        )

    _render_contact_form(formspree_form_id)


if __name__ == "__main__":
    main()
