import json

import pytest

from packages.agent_core.errors import ModelResponseError
from packages.agent_core.groq_gateway import (
    model_options,
    parse_model_plan,
    response_format_for_model,
)


def test_parse_ok_plan():
    plan = parse_model_plan(
        json.dumps(
            {
                "status": "ok",
                "message": "Done",
                "code": "result = df.head()",
                "output_kind": "dataframe",
            }
        )
    )
    assert plan.status == "ok"
    assert plan.code == "result = df.head()"


def test_guardrail_marker_is_normalized():
    plan = parse_model_plan("GUARDRAIL_TRIGGERED")
    assert plan.status == "guardrail"
    assert "GUARDRAIL_TRIGGERED" in plan.message
    assert plan.code == ""


def test_missing_code_is_rejected():
    with pytest.raises(ModelResponseError, match="no executable"):
        parse_model_plan(
            '{"status":"ok","message":"x","code":"","output_kind":"auto"}'
        )


def test_gpt_oss_uses_strict_json_schema_and_hidden_low_reasoning():
    response_format = response_format_for_model("openai/gpt-oss-20b")
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert model_options("openai/gpt-oss-20b") == {
        "include_reasoning": False,
        "reasoning_effort": "low",
    }


def test_configurable_non_strict_model_uses_json_object_mode():
    assert response_format_for_model("some/other-model") == {"type": "json_object"}
    assert model_options("some/other-model") == {}
