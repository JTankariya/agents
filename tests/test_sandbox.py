import json

import pandas as pd
import pytest

from packages.agent_core.errors import SandboxExecutionError, SandboxValidationError
from packages.agent_core.sandbox import SandboxPolicy, execute_safely


@pytest.fixture
def policy() -> SandboxPolicy:
    return SandboxPolicy(timeout_seconds=8.0, extra_memory_bytes=384 * 1024 * 1024)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["A", "A", "B"],
            "value": [1, 2, 4],
            "other": [2.0, 4.0, 8.0],
        }
    )


def test_safe_dataframe_analysis_executes(policy: SandboxPolicy, frame: pd.DataFrame):
    output = execute_safely(
        'result = df.groupby("category", as_index=False)["value"].sum()',
        frame,
        policy,
    )
    assert isinstance(output.result, pd.DataFrame)
    assert output.result["value"].tolist() == [3, 4]


def test_import_is_blocked(policy: SandboxPolicy, frame: pd.DataFrame):
    with pytest.raises(SandboxValidationError, match="Import"):
        execute_safely("import os\nresult = 1", frame, policy)


def test_file_write_is_blocked(policy: SandboxPolicy, frame: pd.DataFrame):
    with pytest.raises(SandboxValidationError, match="to_csv"):
        execute_safely('result = df.to_csv("/tmp/leak.csv")', frame, policy)


def test_dunder_escape_is_blocked(policy: SandboxPolicy, frame: pd.DataFrame):
    with pytest.raises(SandboxValidationError, match="Blocked attribute"):
        execute_safely("result = df.__class__", frame, policy)


def test_loop_is_blocked(policy: SandboxPolicy, frame: pd.DataFrame):
    with pytest.raises(SandboxValidationError, match="For"):
        execute_safely("result = 0\nfor x in [1]:\n    result = x", frame, policy)


def test_matplotlib_figure_executes_and_returns(policy: SandboxPolicy, frame: pd.DataFrame):
    output = execute_safely(
        'fig, ax = plt.subplots()\nax.bar(df["category"], df["value"])\nresult = df.head(2)',
        frame,
        policy,
    )
    assert output.figure_kind == "matplotlib_png"
    assert isinstance(output.figure, bytes)
    assert output.figure.startswith(b"\x89PNG")


def test_plotly_figure_executes_and_returns(policy: SandboxPolicy, frame: pd.DataFrame):
    output = execute_safely(
        'fig = px.bar(df, x="category", y="value")\nresult = df.head(1)',
        frame,
        policy,
    )
    assert output.figure_kind == "plotly_json"
    document = json.loads(output.figure)
    assert document["data"]


def test_transformation_uses_a_copy_of_input_dataframe(
    policy: SandboxPolicy,
    frame: pd.DataFrame,
):
    original = frame.copy(deep=True)
    output = execute_safely(
        'updated_df = df.copy()\nupdated_df["value"] = updated_df["value"] * 10\nresult = updated_df',
        frame,
        policy,
    )
    pd.testing.assert_frame_equal(frame, original)
    assert output.updated_df is not None
    assert output.updated_df["value"].tolist() == [10, 20, 40]


def test_pandas_os_escape_is_blocked(policy: SandboxPolicy, frame: pd.DataFrame):
    with pytest.raises(SandboxValidationError):
        execute_safely("result = pd.io.common.os", frame, policy)


def test_image_file_read_is_blocked(policy: SandboxPolicy, frame: pd.DataFrame):
    with pytest.raises(SandboxValidationError, match="imread"):
        execute_safely('result = plt.imread("/etc/passwd")', frame, policy)


def test_execution_error_redacts_quoted_cell_value(policy: SandboxPolicy):
    secret = "TOP_SECRET_CELL_837461"
    df = pd.DataFrame({"secret": [secret]})
    with pytest.raises(SandboxExecutionError) as error:
        execute_safely('result = int(df["secret"].iloc[0])', df, policy)
    assert secret not in str(error.value)
    assert "ValueError" in str(error.value)


def test_higher_order_file_read_callable_is_blocked(
    policy: SandboxPolicy,
    frame: pd.DataFrame,
):
    with pytest.raises(SandboxValidationError):
        execute_safely("result = df.apply(pd.read_csv)", frame, policy)


def test_internal_pandas_file_handle_escape_is_blocked(
    policy: SandboxPolicy,
    frame: pd.DataFrame,
):
    with pytest.raises(SandboxValidationError):
        execute_safely("result = pd.io.common.get_handle", frame, policy)


def test_trusted_module_cannot_be_aliased(policy: SandboxPolicy, frame: pd.DataFrame):
    with pytest.raises(SandboxValidationError, match="cannot be aliased"):
        execute_safely("helper = pd\nresult = 1", frame, policy)


def test_attribute_assignment_is_blocked(policy: SandboxPolicy, frame: pd.DataFrame):
    with pytest.raises(SandboxValidationError, match="Attribute mutation"):
        execute_safely("df.columns = [\"a\", \"b\", \"c\"]\nresult = df", frame, policy)


def test_common_correlation_analysis_executes(policy: SandboxPolicy, frame: pd.DataFrame):
    output = execute_safely(
        'numeric = df.select_dtypes(include="number")\nresult = numeric.corr()',
        frame,
        policy,
    )
    assert isinstance(output.result, pd.DataFrame)
    assert output.result.loc["value", "other"] == pytest.approx(1.0)


def test_nested_non_data_object_output_is_rejected(
    policy: SandboxPolicy,
    frame: pd.DataFrame,
):
    with pytest.raises(SandboxExecutionError, match="Unsupported value type"):
        execute_safely('result = {"unsafe": df.groupby}', frame, policy)


def test_matplotlib_is_returned_as_inert_png_bytes(
    policy: SandboxPolicy,
    frame: pd.DataFrame,
):
    output = execute_safely(
        'fig, ax = plt.subplots()\nax.plot(df["value"])\nresult = "ok"',
        frame,
        policy,
    )
    assert isinstance(output.figure, bytes)
    assert output.figure_kind == "matplotlib_png"


def test_matplotlib_canvas_file_escape_is_blocked(
    policy: SandboxPolicy,
    frame: pd.DataFrame,
):
    with pytest.raises(SandboxValidationError, match="canvas"):
        execute_safely(
            'fig, ax = plt.subplots()\nresult = fig.canvas',
            frame,
            policy,
        )


def test_dataframe_attrs_are_removed_before_output_crosses_boundary(
    policy: SandboxPolicy,
    frame: pd.DataFrame,
):
    output = execute_safely(
        'result = df.copy()\nresult.attrs["private"] = "metadata"',
        frame,
        policy,
    )
    assert output.result.attrs == {}


def test_unsafe_index_metadata_is_rejected(policy: SandboxPolicy, frame: pd.DataFrame):
    code = "result = df.copy().set_axis(pd.Index(df.index, name=df.groupby), axis=0)"
    with pytest.raises(SandboxExecutionError, match="Unsupported value type"):
        execute_safely(code, frame, policy)


def test_plotly_external_resource_is_rejected_without_exposing_cell_value(
    policy: SandboxPolicy,
):
    secret_url = "https://private.example.invalid/raw-value"
    df = pd.DataFrame({"x": [1], "y": [2], "label": [secret_url]})
    code = 'fig = px.scatter(df, x="x", y="y", title=df["label"].iloc[0])\nresult = "chart"'
    with pytest.raises(SandboxExecutionError) as error:
        execute_safely(code, df, policy)
    assert secret_url not in str(error.value)
    assert "External URLs" in str(error.value)
