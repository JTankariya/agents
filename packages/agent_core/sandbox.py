"""Defense-in-depth execution for LLM-generated pandas code.

The code uses native exec(), but only after AST validation and inside a short-lived
child process. This is a portfolio-grade boundary, not a formal hostile multi-tenant
sandbox.
"""

from __future__ import annotations

import ast
import io
import json
import math
import multiprocessing as mp
import pickle
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas.api import types as ptypes

from .contracts import SandboxOutput
from .errors import (
    SandboxError,
    SandboxExecutionError,
    SandboxTimeoutError,
    SandboxValidationError,
)


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    timeout_seconds: float = 10.0
    extra_memory_bytes: int = 512 * 1024 * 1024
    max_code_chars: int = 5_000
    max_ast_nodes: int = 600
    max_result_bytes: int = 96 * 1024 * 1024
    max_figure_bytes: int = 12 * 1024 * 1024
    max_text_chars: int = 100_000
    max_container_items: int = 20_000
    max_rows: int = 50_000
    max_columns: int = 100


_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

_OUTPUT_NAMES = {"result", "fig", "updated_df", "message"}
_TRUSTED_MODULE_NAMES = {"pd", "np", "plt", "px"}
_RESERVED_NAMES = _TRUSTED_MODULE_NAMES | {"df", "__builtins__"}
_BLOCKED_NAMES = {
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "print",
    "quit",
    "setattr",
    "vars",
}
_BLOCKED_ATTRIBUTE_TOKENS = {
    "builtins",
    "canvas",
    "clipboard",
    "ctypes",
    "ctypeslib",
    "environ",
    "environment",
    "fork",
    "glob",
    "hdfstore",
    "importlib",
    "inspect",
    "manager",
    "marshal",
    "matplotlib",
    "memmap",
    "os",
    "pathlib",
    "pickle",
    "popen",
    "requests",
    "reset_option",
    "set_option",
    "seterr",
    "shelve",
    "shutil",
    "socket",
    "spawn",
    "sqlite3",
    "subprocess",
    "sys",
    "tempfile",
    "urllib",
    "webbrowser",
    "walk",
}
_BLOCKED_CALL_ATTRIBUTES = {
    "connect",
    "dump",
    "dumps",
    "eval",
    "exec",
    "fromfile",
    "genfromtxt",
    "get_handle",
    "imread",
    "imsave",
    "load",
    "loads",
    "loadtxt",
    "open",
    "query",
    "read",
    "read_clipboard",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_fwf",
    "read_gbq",
    "read_hdf",
    "read_html",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_sas",
    "read_spss",
    "read_sql",
    "read_stata",
    "read_table",
    "read_xml",
    "request",
    "save",
    "savefig",
    "savetxt",
    "savez",
    "savez_compressed",
    "show",
    "system",
    "to_clipboard",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_gbq",
    "to_hdf",
    "to_html",
    "to_json",
    "to_latex",
    "to_orc",
    "to_parquet",
    "to_pickle",
    "to_sql",
    "to_stata",
    "to_xml",
    "tofile",
    "urlopen",
    "write",
    "write_html",
    "write_image",
    "writelines",
}
_SAFE_TO_METHODS = {
    "to_datetime",
    "to_dict",
    "to_frame",
    "to_list",
    "to_markdown",
    "to_numpy",
    "to_numeric",
    "to_period",
    "to_string",
    "to_timedelta",
    "to_timestamp",
    "tolist",
}
_FORBIDDEN_NODE_TYPES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.DictComp,
    ast.For,
    ast.FunctionDef,
    ast.GeneratorExp,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.ListComp,
    ast.Match,
    ast.NamedExpr,
    ast.Nonlocal,
    ast.Raise,
    ast.SetComp,
    ast.Try,
    ast.While,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)

_ALLOWED_MODULE_PATHS = {
    # pandas constructors, reshaping helpers, safe conversion, and dtype checks
    "pd.DataFrame",
    "pd.DataFrame.from_dict",
    "pd.DataFrame.from_records",
    "pd.Series",
    "pd.Index",
    "pd.MultiIndex",
    "pd.RangeIndex",
    "pd.Categorical",
    "pd.Grouper",
    "pd.NamedAgg",
    "pd.Timestamp",
    "pd.Timedelta",
    "pd.Interval",
    "pd.Period",
    "pd.NA",
    "pd.NaT",
    "pd.array",
    "pd.concat",
    "pd.merge",
    "pd.crosstab",
    "pd.pivot_table",
    "pd.cut",
    "pd.qcut",
    "pd.get_dummies",
    "pd.to_datetime",
    "pd.to_numeric",
    "pd.to_timedelta",
    "pd.isna",
    "pd.isnull",
    "pd.notna",
    "pd.notnull",
    "pd.unique",
    "pd.factorize",
    "pd.api.types.is_bool_dtype",
    "pd.api.types.is_datetime64_any_dtype",
    "pd.api.types.is_numeric_dtype",
    "pd.api.types.is_string_dtype",
    # numpy math and array helpers; no I/O, random, loaders, or object access
    "np.ndarray",
    "np.number",
    "np.integer",
    "np.floating",
    "np.bool_",
    "np.array",
    "np.asarray",
    "np.abs",
    "np.mean",
    "np.median",
    "np.std",
    "np.var",
    "np.sum",
    "np.min",
    "np.max",
    "np.nanmean",
    "np.nanmedian",
    "np.nanstd",
    "np.nanvar",
    "np.nansum",
    "np.nanmin",
    "np.nanmax",
    "np.percentile",
    "np.quantile",
    "np.nanpercentile",
    "np.nanquantile",
    "np.where",
    "np.select",
    "np.clip",
    "np.round",
    "np.around",
    "np.sqrt",
    "np.log",
    "np.log1p",
    "np.log10",
    "np.exp",
    "np.expm1",
    "np.power",
    "np.isnan",
    "np.isfinite",
    "np.isinf",
    "np.corrcoef",
    "np.cov",
    "np.histogram",
    "np.histogram2d",
    "np.unique",
    "np.sort",
    "np.argsort",
    "np.argmax",
    "np.argmin",
    "np.ravel",
    "np.reshape",
    "np.transpose",
    "np.triu",
    "np.tril",
    "np.ceil",
    "np.floor",
    "np.polyfit",
    "np.polyval",
    "np.arange",
    "np.linspace",
    "np.zeros",
    "np.ones",
    "np.full",
    "np.empty",
    "np.repeat",
    "np.tile",
    "np.datetime64",
    "np.timedelta64",
    "np.dtype",
    "np.nan",
    "np.inf",
    "np.pi",
    "np.e",
    "np.linalg.norm",
    "np.linalg.det",
    "np.linalg.eig",
    "np.linalg.eigvals",
    # Matplotlib construction only; trusted code serializes figures.
    "plt.subplots",
    "plt.figure",
    "plt.tight_layout",
    "plt.colorbar",
    "plt.title",
    "plt.xlabel",
    "plt.ylabel",
    "plt.xticks",
    "plt.yticks",
    "plt.gca",
    "plt.gcf",
    # Plotly Express constructors only.
    "px.area",
    "px.bar",
    "px.box",
    "px.choropleth",
    "px.density_contour",
    "px.density_heatmap",
    "px.ecdf",
    "px.funnel",
    "px.histogram",
    "px.imshow",
    "px.line",
    "px.pie",
    "px.scatter",
    "px.scatter_3d",
    "px.scatter_matrix",
    "px.strip",
    "px.sunburst",
    "px.treemap",
    "px.violin",
}


def _dotted_path(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _root_name(node: ast.AST) -> str | None:
    current = node
    while isinstance(current, (ast.Attribute, ast.Subscript)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


def _contains_trusted_module_reference(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in _TRUSTED_MODULE_NAMES:
            return True
    return False


class _PolicyVisitor(ast.NodeVisitor):
    def __init__(self, policy: SandboxPolicy) -> None:
        self.policy = policy
        self.node_count = 0

    def generic_visit(self, node: ast.AST) -> None:
        self.node_count += 1
        if self.node_count > self.policy.max_ast_nodes:
            raise SandboxValidationError("Generated code is too complex for the sandbox.")
        if isinstance(node, _FORBIDDEN_NODE_TYPES):
            raise SandboxValidationError(
                f"Generated code uses blocked syntax: {type(node).__name__}."
            )
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__") or node.id in _BLOCKED_NAMES:
            raise SandboxValidationError(f"Blocked name used: {node.id}.")
        if isinstance(node.ctx, ast.Store) and node.id in _RESERVED_NAMES:
            raise SandboxValidationError(f"Trusted name cannot be reassigned: {node.id}.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr.lower()
        if node.attr.startswith("__") or attr in _BLOCKED_ATTRIBUTE_TOKENS:
            raise SandboxValidationError(f"Blocked attribute used: {node.attr}.")
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            raise SandboxValidationError("Attribute mutation is blocked.")
        root = _root_name(node)
        if root in _TRUSTED_MODULE_NAMES:
            path = _dotted_path(node)
            if path not in _ALLOWED_MODULE_PATHS:
                raise SandboxValidationError(f"Module API is not allowed: {path or root}.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if any(keyword.arg is None for keyword in node.keywords):
            raise SandboxValidationError("Dictionary argument expansion is blocked.")
        blocked_keywords = {"backend", "engine", "storage_options"}
        if any(keyword.arg in blocked_keywords for keyword in node.keywords):
            raise SandboxValidationError("Dynamic backends and storage engines are blocked.")
        if isinstance(node.func, ast.Name):
            if node.func.id not in _SAFE_BUILTINS:
                raise SandboxValidationError(
                    f"Calling local or unknown callable is blocked: {node.func.id}."
                )
        elif isinstance(node.func, ast.Attribute):
            attr = node.func.attr.lower()
            if attr in _BLOCKED_CALL_ATTRIBUTES or (
                attr.startswith("to_") and attr not in _SAFE_TO_METHODS
            ):
                raise SandboxValidationError(f"Blocked method call: {node.func.attr}.")
            root = _root_name(node.func)
            if root in _TRUSTED_MODULE_NAMES:
                path = _dotted_path(node.func)
                if path not in _ALLOWED_MODULE_PATHS:
                    raise SandboxValidationError(
                        f"Module API call is not allowed: {path or root}."
                    )
        else:
            raise SandboxValidationError("Dynamic callable expressions are blocked.")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        alias_shapes = (ast.Name, ast.Attribute, ast.List, ast.Tuple, ast.Set, ast.Dict)
        if isinstance(node.value, alias_shapes) and _contains_trusted_module_reference(node.value):
            raise SandboxValidationError("Trusted modules and their callables cannot be aliased.")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and isinstance(node.value, (ast.Name, ast.Attribute)):
            if _contains_trusted_module_reference(node.value):
                raise SandboxValidationError(
                    "Trusted modules and their callables cannot be aliased."
                )
        self.generic_visit(node)


def _attach_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            setattr(child, "_sandbox_parent", parent)


def validate_code(code: str, policy: SandboxPolicy) -> ast.Module:
    if not code.strip():
        raise SandboxValidationError("Generated code is empty.")
    if len(code) > policy.max_code_chars:
        raise SandboxValidationError("Generated code exceeds the source-size limit.")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxValidationError(
            f"Generated code has invalid syntax near line {exc.lineno or '?'}."
        ) from exc
    _attach_parents(tree)
    _PolicyVisitor(policy).visit(tree)

    assigned_names: set[str] = set()

    def collect_names(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            assigned_names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                collect_names(element)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                collect_names(target)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            collect_names(node.target)
    if not assigned_names.intersection(_OUTPUT_NAMES):
        raise SandboxValidationError(
            "Generated code must assign result, fig, updated_df, or message."
        )
    return tree


def _generated_line_number(exc: BaseException) -> int | None:
    traceback = exc.__traceback__
    generated_line: int | None = None
    while traceback is not None:
        if traceback.tb_frame.f_code.co_filename == "<agent-generated>":
            generated_line = traceback.tb_lineno
        traceback = traceback.tb_next
    return generated_line


def _sanitize_exception_message(exc: BaseException) -> str:
    """Create a repair hint without forwarding third-party exception values."""

    if isinstance(exc, SandboxError):
        message = re.sub(r"\s+", " ", str(exc)).strip()
        message = re.sub(r"(?:[A-Za-z]:)?[/\\][^\s]+", "<path-redacted>", message)
        return f"{type(exc).__name__}: {(message or 'Sandbox rule violated.')[:500]}"

    summaries: tuple[tuple[type[BaseException], str], ...] = (
        (KeyError, "A referenced row label or column label was not found."),
        (IndexError, "A row or column position was outside the available bounds."),
        (TypeError, "An operation used incompatible data types or arguments."),
        (ValueError, "An operation received an invalid value or conversion."),
        (AttributeError, "An unavailable attribute or method was used."),
        (ZeroDivisionError, "An arithmetic operation attempted division by zero."),
        (OverflowError, "An arithmetic operation exceeded numeric limits."),
        (MemoryError, "The generated operation exceeded the memory allowance."),
    )
    summary = "The generated program raised an execution error."
    for exception_type, candidate in summaries:
        if isinstance(exc, exception_type):
            summary = candidate
            break
    line_number = _generated_line_number(exc)
    location = f" at generated line {line_number}" if line_number is not None else ""
    return f"{type(exc).__name__}{location}: {summary}"


def _current_vms_bytes() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _apply_process_limits(policy: SandboxPolicy) -> None:
    try:
        import resource

        cpu_seconds = max(1, int(math.ceil(policy.timeout_seconds)))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        current_vms = _current_vms_bytes()
        if current_vms is not None:
            address_space_limit = current_vms + policy.extra_memory_bytes
            resource.setrlimit(
                resource.RLIMIT_AS,
                (address_space_limit, address_space_limit),
            )
    except (ImportError, OSError, ValueError):
        pass


def _serialized_size(value: Any) -> int:
    try:
        return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception as exc:
        raise SandboxExecutionError("The analysis output cannot be serialized safely.") from exc


def _normalize_safe_value(
    value: Any,
    policy: SandboxPolicy,
    *,
    depth: int = 0,
    budget: list[int] | None = None,
) -> Any:
    if depth > 8:
        raise SandboxExecutionError("The analysis output is nested too deeply.")
    if budget is None:
        budget = [0]
    budget[0] += 1
    if budget[0] > policy.max_container_items:
        raise SandboxExecutionError("The analysis output contains too many items.")

    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _normalize_safe_value(value.item(), policy, depth=depth, budget=budget)
    if isinstance(value, (pd.Timestamp, pd.Timedelta, pd.Period, pd.Interval)):
        return str(value)
    if isinstance(value, (datetime, date, time, timedelta)):
        return value.isoformat() if hasattr(value, "isoformat") else str(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, complex):
        return str(value)
    if isinstance(value, str):
        if len(value) > policy.max_text_chars:
            raise SandboxExecutionError("A text result exceeds the display-size limit.")
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, np.ndarray):
        if value.size > policy.max_container_items:
            raise SandboxExecutionError("The generated array is too large to display.")
        return _normalize_safe_value(value.tolist(), policy, depth=depth + 1, budget=budget)
    if isinstance(value, pd.Index):
        if len(value) > policy.max_container_items:
            raise SandboxExecutionError("The generated index is too large to display.")
        return _normalize_safe_value(value.tolist(), policy, depth=depth + 1, budget=budget)
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            safe_key = _normalize_safe_value(key, policy, depth=depth + 1, budget=budget)
            normalized[str(safe_key)] = _normalize_safe_value(
                item, policy, depth=depth + 1, budget=budget
            )
        return normalized
    if isinstance(value, list):
        return [
            _normalize_safe_value(item, policy, depth=depth + 1, budget=budget)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _normalize_safe_value(item, policy, depth=depth + 1, budget=budget)
            for item in value
        )
    raise SandboxExecutionError(
        f"Unsupported value type in output: {type(value).__name__}."
    )


def _validate_axis(axis: pd.Index, policy: SandboxPolicy) -> None:
    names = list(axis.names) if isinstance(axis, pd.MultiIndex) else [axis.name]
    for name in names:
        _normalize_safe_value(name, policy, budget=[0])
    for value in axis.tolist():
        _normalize_safe_value(value, policy, budget=[0])


def _validate_dataframe(value: pd.DataFrame, policy: SandboxPolicy) -> None:
    if type(value) is not pd.DataFrame:
        raise SandboxExecutionError("DataFrame subclasses are not allowed as output.")
    value.attrs.clear()
    if value.shape[0] > policy.max_rows or value.shape[1] > policy.max_columns:
        raise SandboxExecutionError("The generated DataFrame exceeds the output shape limit.")
    _validate_axis(value.index, policy)
    _validate_axis(value.columns, policy)
    for column in value.columns:
        series = value[column]
        if (
            ptypes.is_object_dtype(series.dtype)
            or ptypes.is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            for cell in series.array:
                _normalize_safe_value(cell, policy, budget=[0])


def _validate_series(value: pd.Series, policy: SandboxPolicy) -> None:
    if type(value) is not pd.Series:
        raise SandboxExecutionError("Series subclasses are not allowed as output.")
    value.attrs.clear()
    _normalize_safe_value(value.name, policy, budget=[0])
    if len(value) > policy.max_rows:
        raise SandboxExecutionError("The generated Series is too large to return.")
    _validate_axis(value.index, policy)
    if (
        ptypes.is_object_dtype(value.dtype)
        or ptypes.is_string_dtype(value.dtype)
        or isinstance(value.dtype, pd.CategoricalDtype)
    ):
        for cell in value.array:
            _normalize_safe_value(cell, policy, budget=[0])


_NETWORK_VALUE_PATTERN = re.compile(
    r"(?i)(?:https?|ftp|file|data|javascript|ws|wss):|(?:^|[\s\"\'])//"
)


def _validate_plotly_json(payload: str, policy: SandboxPolicy) -> None:
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SandboxExecutionError("The Plotly chart returned invalid JSON.") from exc
    budget = [0]

    def walk(value: Any, depth: int = 0) -> None:
        if depth > 20:
            raise SandboxExecutionError("The Plotly chart is nested too deeply.")
        budget[0] += 1
        if budget[0] > policy.max_container_items * 10:
            raise SandboxExecutionError("The Plotly chart contains too many elements.")
        if isinstance(value, str):
            lowered = value.lower()
            if _NETWORK_VALUE_PATTERN.search(value) or "<script" in lowered:
                raise SandboxExecutionError(
                    "External URLs and executable markup are blocked in chart output."
                )
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(str(key), depth + 1)
                walk(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                walk(item, depth + 1)

    walk(document)


def _serialize_figure(figure: Any, policy: SandboxPolicy) -> tuple[bytes | str | None, str]:
    if figure is None:
        return None, ""
    if isinstance(figure, matplotlib.axes.Axes):
        figure = figure.figure
    if isinstance(figure, matplotlib.figure.Figure):
        buffer = io.BytesIO()
        try:
            figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        except Exception as exc:
            raise SandboxExecutionError("The Matplotlib chart could not be rendered.") from exc
        payload = buffer.getvalue()
        if len(payload) > policy.max_figure_bytes:
            raise SandboxExecutionError("The rendered chart exceeds the image-size limit.")
        return payload, "matplotlib_png"
    if isinstance(figure, go.Figure):
        try:
            payload = figure.to_json(validate=True)
        except Exception as exc:
            raise SandboxExecutionError("The Plotly chart could not be serialized.") from exc
        if len(payload.encode("utf-8")) > policy.max_figure_bytes:
            raise SandboxExecutionError("The rendered chart exceeds the JSON-size limit.")
        _validate_plotly_json(payload, policy)
        return payload, "plotly_json"
    raise SandboxExecutionError(f"Unsupported figure type: {type(figure).__name__}.")


def _validate_output(namespace: dict[str, Any], policy: SandboxPolicy) -> SandboxOutput:
    result = namespace.get("result")
    figure = namespace.get("fig")
    updated_df = namespace.get("updated_df")
    message = namespace.get("message", "")

    if isinstance(result, pd.DataFrame):
        _validate_dataframe(result, policy)
    elif isinstance(result, pd.Series):
        _validate_series(result, policy)
    elif result is not None:
        result = _normalize_safe_value(result, policy)

    if updated_df is not None:
        if not isinstance(updated_df, pd.DataFrame):
            raise SandboxExecutionError("updated_df must be a pandas DataFrame.")
        _validate_dataframe(updated_df, policy)

    if not isinstance(message, str):
        raise SandboxExecutionError("message must be a plain string.")
    if len(message) > 1_000:
        message = message[:1_000]

    figure_payload, figure_kind = _serialize_figure(figure, policy)
    output = SandboxOutput(
        result=result,
        figure=figure_payload,
        figure_kind=figure_kind,  # type: ignore[arg-type]
        updated_df=updated_df,
        message=message,
    )
    if _serialized_size(output) > policy.max_result_bytes:
        raise SandboxExecutionError("The combined analysis output is too large to return.")
    return output


def _sandbox_worker(
    child_connection: Any,
    code: str,
    dataframe: pd.DataFrame,
    policy: SandboxPolicy,
) -> None:
    try:
        _apply_process_limits(policy)
        tree = validate_code(code, policy)
        namespace: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            "pd": pd,
            "np": np,
            "df": dataframe.copy(deep=True),
            "plt": plt,
            "px": px,
        }
        compiled = compile(tree, filename="<agent-generated>", mode="exec")
        exec(compiled, namespace, namespace)
        output = _validate_output(namespace, policy)
        child_connection.send({"ok": True, "output": output})
    except BaseException as exc:
        try:
            child_connection.send({"ok": False, "error": _sanitize_exception_message(exc)})
        except BaseException:
            pass
    finally:
        try:
            child_connection.close()
        except OSError:
            pass
        plt.close("all")


def execute_safely(
    code: str,
    dataframe: pd.DataFrame,
    policy: SandboxPolicy,
) -> SandboxOutput:
    """Validate and execute generated code in a disposable process."""

    validate_code(code, policy)
    process: mp.Process | None = None
    try:
        context = mp.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_sandbox_worker,
            args=(child_connection, code, dataframe, policy),
            daemon=True,
        )
        process.start()
        child_connection.close()
    except Exception as exc:
        for connection_name in ("parent_connection", "child_connection"):
            connection = locals().get(connection_name)
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
        raise SandboxExecutionError(
            "The isolated analysis process could not be started."
        ) from exc

    payload: Any = None
    try:
        try:
            ready = parent_connection.poll(policy.timeout_seconds)
        except (OSError, EOFError) as exc:
            raise SandboxExecutionError(
                "Communication with the isolated analysis process failed."
            ) from exc
        if not ready:
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1.0)
            raise SandboxTimeoutError(
                "The generated analysis exceeded the execution time limit."
            )
        try:
            payload = parent_connection.recv()
        except (EOFError, OSError) as exc:
            raise SandboxExecutionError(
                "The isolated analysis process ended unexpectedly."
            ) from exc
    finally:
        parent_connection.close()
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=1.0)
        try:
            process.close()
        except ValueError:
            pass

    if not isinstance(payload, dict) or not payload.get("ok"):
        safe_error = (
            str(payload.get("error", "Isolated execution failed."))
            if isinstance(payload, dict)
            else "Isolated execution failed."
        )
        raise SandboxExecutionError(safe_error)
    output = payload.get("output")
    if not isinstance(output, SandboxOutput):
        raise SandboxExecutionError("The isolated analysis returned an invalid payload.")
    return output
