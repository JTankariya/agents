import io

import pandas as pd
import pytest

from packages.agent_core.config import AppSettings
from packages.agent_core.data_profile import build_dataset_profile, load_csv_bytes
from packages.agent_core.errors import DatasetError


def test_profile_never_contains_raw_values():
    df = pd.DataFrame(
        {
            "customer": ["PRIVACY_SENTINEL_A", "PRIVACY_SENTINEL_B"],
            "revenue": [123456789, 987654321],
        }
    )
    context = build_dataset_profile(df).as_prompt_text()
    assert "customer" in context
    assert "revenue" in context
    assert "PRIVACY_SENTINEL" not in context
    assert "123456789" not in context
    assert "<text>" in context
    assert "<integer>" in context


def test_loader_truncates_without_reading_full_dataset():
    settings = AppSettings(max_rows=3)
    payload = b"a,b\n1,x\n2,y\n3,z\n4,q\n5,r\n"
    loaded = load_csv_bytes(payload, settings)
    assert loaded.truncated is True
    assert len(loaded.dataframe) == 3
    assert loaded.dataframe["a"].tolist() == [1, 2, 3]


def test_structural_sample_caps_wide_datasets_but_lists_all_columns():
    df = pd.DataFrame({f"col_{index}": [index] for index in range(25)})
    profile = build_dataset_profile(df)
    context = profile.as_prompt_text()
    assert profile.redacted_sample_column_count == 20
    assert '"name": "col_24"' in context
    assert "5 additional columns" in context


def test_loader_rejects_oversized_schema_header():
    settings = AppSettings(max_total_column_name_chars=10)
    payload = io.StringIO()
    payload.write("very_long_one,very_long_two\n1,2\n")
    with pytest.raises(DatasetError, match="combined CSV header"):
        load_csv_bytes(payload.getvalue().encode(), settings)
