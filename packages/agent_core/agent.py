"""One-retry agent orchestration."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from .contracts import AgentRunResult, ModelPlan, TraceEvent
from .data_profile import build_dataset_profile
from .errors import AgentPlaygroundError, GatewayError, SandboxError
from .sandbox import SandboxPolicy, execute_safely


TraceCallback = Callable[[TraceEvent], None]


class DataAnalystAgent:
    """Generate, validate, execute, and repair one pandas analysis."""

    def __init__(self, gateway: object, sandbox_policy: SandboxPolicy) -> None:
        self.gateway = gateway
        self.sandbox_policy = sandbox_policy

    def run(
        self,
        *,
        user_request: str,
        dataframe: pd.DataFrame,
        on_trace: TraceCallback | None = None,
    ) -> AgentRunResult:
        trace: list[TraceEvent] = []

        def emit(kind: str, label: str, detail: str = "", code: str = "") -> None:
            event = TraceEvent(kind=kind, label=label, detail=detail, code=code)  # type: ignore[arg-type]
            trace.append(event)
            if on_trace is not None:
                on_trace(event)

        emit("step", "Extracting schema", "Building a privacy-preserving dataset profile.")
        try:
            dataset_context = build_dataset_profile(dataframe).as_prompt_text()
        except Exception:
            message = "The dataset schema could not be prepared safely."
            emit("error", "Schema extraction failed", message)
            return AgentRunResult(
                status="failed",
                message=message,
                attempts=0,
                errors=[message],
                trace=trace,
            )

        try:
            emit(
                "step",
                "Generating pandas script",
                "Calling Groq with your request and approved metadata; no dataset rows.",
            )
            plan: ModelPlan = self.gateway.generate(user_request, dataset_context)  # type: ignore[attr-defined]
        except AgentPlaygroundError as exc:
            emit("error", "Generation failed", str(exc))
            return AgentRunResult(
                status="failed",
                message=str(exc),
                attempts=0,
                errors=[str(exc)],
                trace=trace,
            )
        except Exception:
            safe = "The model gateway failed before code generation."
            emit("error", "Generation failed", safe)
            return AgentRunResult(
                status="failed",
                message=safe,
                attempts=0,
                errors=[safe],
                trace=trace,
            )

        if plan.status == "guardrail":
            emit("warning", "Scope guardrail", plan.message)
            return AgentRunResult(
                status="guardrail",
                message=plan.message,
                plan=plan,
                attempts=0,
                trace=trace,
            )

        emit("code", "Generated code", "Review the code before sandbox execution.", plan.code)

        errors: list[str] = []
        current_plan = plan
        attempts_made = 0
        for attempt in (1, 2):
            attempts_made = attempt
            try:
                emit(
                    "step",
                    "Sandboxed execution",
                    f"Validating and executing attempt {attempt} of 2.",
                )
                output = execute_safely(current_plan.code, dataframe, self.sandbox_policy)
                emit("success", "Execution complete", f"Attempt {attempt} succeeded.")
                return AgentRunResult(
                    status="success",
                    message=output.message or current_plan.message,
                    plan=current_plan,
                    output=output,
                    attempts=attempt,
                    errors=errors,
                    trace=trace,
                )
            except SandboxError as exc:
                safe_error = str(exc)
            except Exception:
                safe_error = "The isolated analysis failed unexpectedly."

            errors.append(safe_error)
            emit("warning", "Execution issue", safe_error)
            if attempt == 2:
                break

            try:
                emit(
                    "step",
                    "Self-healing retry",
                    "Sending your request, code, approved metadata, and a sanitized error to Groq.",
                )
                current_plan = self.gateway.repair(  # type: ignore[attr-defined]
                    user_request,
                    dataset_context,
                    current_plan.code,
                    safe_error,
                )
            except GatewayError as repair_exc:
                errors.append(str(repair_exc))
                emit("error", "Repair failed", str(repair_exc))
                break
            except Exception:
                repair_message = "The model gateway failed during the one allowed repair."
                errors.append(repair_message)
                emit("error", "Repair failed", repair_message)
                break

            if current_plan.status == "guardrail":
                emit("warning", "Scope guardrail", current_plan.message)
                return AgentRunResult(
                    status="guardrail",
                    message=current_plan.message,
                    plan=current_plan,
                    attempts=attempt,
                    errors=errors,
                    trace=trace,
                )
            emit(
                "code",
                "Corrected code",
                "The model supplied one corrected script.",
                current_plan.code,
            )

        message = (
            "The agent could not complete this request safely after one correction. "
            "Try a more specific analysis using exact column names."
        )
        emit("error", "Execution halted", message)
        return AgentRunResult(
            status="failed",
            message=message,
            plan=current_plan,
            attempts=attempts_made,
            errors=errors,
            trace=trace,
        )
