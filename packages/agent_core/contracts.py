"""Typed contracts shared by the agent, sandbox, and UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


PlanStatus = Literal["ok", "guardrail"]
RunStatus = Literal["success", "guardrail", "failed"]
OutputKind = Literal["auto", "dataframe", "chart", "text"]
TraceKind = Literal["step", "code", "warning", "error", "success"]
FigureKind = Literal["", "matplotlib_png", "plotly_json"]


@dataclass(frozen=True, slots=True)
class ModelPlan:
    status: PlanStatus
    message: str
    code: str = ""
    output_kind: OutputKind = "auto"


@dataclass(frozen=True, slots=True)
class TraceEvent:
    kind: TraceKind
    label: str
    detail: str = ""
    code: str = ""


@dataclass(slots=True)
class SandboxOutput:
    result: Any = None
    figure: bytes | str | None = None
    figure_kind: FigureKind = ""
    updated_df: pd.DataFrame | None = None
    message: str = ""


@dataclass(slots=True)
class AgentRunResult:
    status: RunStatus
    message: str
    attempts: int
    plan: ModelPlan | None = None
    output: SandboxOutput | None = None
    errors: list[str] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
