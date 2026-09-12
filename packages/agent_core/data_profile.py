"""CSV loading and privacy-preserving dataset context generation."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import pandas as pd
from pandas.api import types as ptypes

from .config import AppSettings
from .errors import DatasetError


_MAX_STRUCTURAL_SAMPLE_COLUMNS = 20


@dataclass(frozen=True, slots=True)
class CsvLoadResult:
    dataframe: pd.DataFrame
    truncated: bool
    encoding: str
    memory_bytes: int


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    row_count: int
    column_count: int
    columns: tuple[dict[str, str], ...]
    redacted_sample_markdown: str
    redacted_sample_column_count: int

    def as_prompt_text(self) -> str:
        """Serialize only approved metadata for the LLM prompt."""

        omitted = self.column_count - self.redacted_sample_column_count
        payload = {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": list(self.columns),
            "redacted_structural_sample_markdown": self.redacted_sample_markdown,
            "structural_sample_note": (
                "Sample cells are type placeholders, not original values. "
                f"The Markdown sample shows the first {self.redacted_sample_column_count} "
                f"columns; {max(0, omitted)} additional columns are described above."
            ),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


def _memory_usage_bytes(df: pd.DataFrame) -> int:
    return int(df.memory_usage(index=True, deep=True).sum())


def _validate_column_names(df: pd.DataFrame, settings: AppSettings) -> None:
    names = [str(column) for column in df.columns]
    if any(len(name) > settings.max_column_name_chars for name in names):
        raise DatasetError(
            "A CSV column name is too long for the safe schema prompt. "
            f"Keep each name within {settings.max_column_name_chars} characters."
        )
    if sum(len(name) for name in names) > settings.max_total_column_name_chars:
        raise DatasetError(
            "The combined CSV header is too large for the free-tier schema prompt. "
            "Shorten the column names or upload a narrower dataset."
        )
    if any(any(ord(character) < 32 for character in name) for name in names):
        raise DatasetError(
            "CSV column names cannot contain tabs, line breaks, or control characters."
        )


def load_csv_bytes(file_bytes: bytes, settings: AppSettings) -> CsvLoadResult:
    """Load at most max_rows + 1 records and fail with safe messages."""

    if not file_bytes:
        raise DatasetError("The uploaded CSV is empty.")
    if len(file_bytes) > settings.max_upload_bytes:
        raise DatasetError("The CSV exceeds the 5 MB upload limit.")
    if b"\x00" in file_bytes[:65_536]:
        raise DatasetError("The upload appears to be a binary file, not a CSV.")

    last_error: Exception | None = None
    dataframe: pd.DataFrame | None = None
    selected_encoding = ""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            dataframe = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding=encoding,
                nrows=settings.max_rows + 1,
                low_memory=False,
            )
            selected_encoding = encoding
            break
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError as exc:
            raise DatasetError("The CSV has no readable columns or rows.") from exc
        except pd.errors.ParserError as exc:
            raise DatasetError(
                "The CSV is malformed. Check delimiters, quoting, and row lengths."
            ) from exc
        except (ValueError, TypeError) as exc:
            raise DatasetError("The CSV could not be parsed safely.") from exc

    if dataframe is None:
        raise DatasetError(
            "The CSV encoding is unsupported. Save it as UTF-8 and upload again."
        ) from last_error
    if dataframe.shape[1] == 0:
        raise DatasetError("The CSV does not contain any columns.")
    if dataframe.shape[1] > settings.max_columns:
        raise DatasetError(
            f"The CSV has too many columns ({dataframe.shape[1]}). "
            f"The safe free-tier limit is {settings.max_columns}."
        )
    _validate_column_names(dataframe, settings)

    truncated = len(dataframe) > settings.max_rows
    if truncated:
        dataframe = dataframe.iloc[: settings.max_rows].copy()

    memory_bytes = _memory_usage_bytes(dataframe)
    if memory_bytes > settings.max_dataframe_bytes:
        raise DatasetError(
            "The parsed dataset expands beyond the in-memory safety limit. "
            "Reduce wide text columns or upload a smaller sample."
        )

    return CsvLoadResult(
        dataframe=dataframe,
        truncated=truncated,
        encoding=selected_encoding,
        memory_bytes=memory_bytes,
    )


def _placeholder_for_dtype(dtype: object) -> str:
    if ptypes.is_bool_dtype(dtype):
        return "<boolean>"
    if ptypes.is_integer_dtype(dtype):
        return "<integer>"
    if ptypes.is_float_dtype(dtype) or ptypes.is_numeric_dtype(dtype):
        return "<number>"
    if ptypes.is_datetime64_any_dtype(dtype):
        return "<datetime>"
    if ptypes.is_timedelta64_dtype(dtype):
        return "<duration>"
    if isinstance(dtype, pd.CategoricalDtype):
        return "<category>"
    return "<text>"


def build_dataset_profile(df: pd.DataFrame) -> DatasetProfile:
    """Build schema context without exposing any cell value to the LLM."""

    columns = tuple(
        {"name": str(column), "dtype": str(dtype)}
        for column, dtype in zip(df.columns, df.dtypes, strict=True)
    )
    sample_columns = list(df.columns[:_MAX_STRUCTURAL_SAMPLE_COLUMNS])
    sample_rows = min(5, len(df))
    redacted_sample = pd.DataFrame(
        {
            str(column): [_placeholder_for_dtype(df[column].dtype)] * sample_rows
            for column in sample_columns
        }
    )
    return DatasetProfile(
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        columns=columns,
        redacted_sample_markdown=redacted_sample.to_markdown(index=False),
        redacted_sample_column_count=len(sample_columns),
    )
