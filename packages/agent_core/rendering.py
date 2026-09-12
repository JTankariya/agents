"""Streamlit rendering helpers for sandbox outputs."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .contracts import SandboxOutput


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def render_output(output: SandboxOutput) -> None:
    if output.message:
        st.info(output.message)

    if output.figure_kind == "matplotlib_png" and isinstance(output.figure, bytes):
        st.image(output.figure, width="stretch")
    elif output.figure_kind == "plotly_json" and isinstance(output.figure, str):
        try:
            figure = go.Figure(json.loads(output.figure))
        except (TypeError, ValueError, json.JSONDecodeError):
            st.warning("The chart output could not be reconstructed safely.")
        else:
            st.plotly_chart(figure, width="stretch")

    result = output.result
    if isinstance(result, pd.DataFrame):
        st.dataframe(result, width="stretch", height=420)
    elif isinstance(result, pd.Series):
        st.dataframe(result.to_frame(), width="stretch", height=420)
    elif isinstance(result, (dict, list, tuple)):
        try:
            st.json(_json_safe(result), expanded=True)
        except (TypeError, ValueError, json.JSONDecodeError):
            st.write(result)
    elif result is not None:
        st.write(_json_safe(result))

    if output.updated_df is not None:
        st.success(
            f"A transformed dataset is ready: {output.updated_df.shape[0]:,} rows × "
            f"{output.updated_df.shape[1]:,} columns."
        )
        csv_bytes = output.updated_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download transformed CSV",
            data=csv_bytes,
            file_name="transformed_dataset.csv",
            mime="text/csv",
            width="stretch",
        )
