"""Official Groq SDK integration with a strict JSON response contract."""

from __future__ import annotations

import json
from typing import Any

from .contracts import ModelPlan
from .errors import GatewayError, GatewayRateLimitError, ModelResponseError
from .prompts import SYSTEM_PROMPT, generation_user_prompt, repair_user_prompt
from .rate_limit import SlidingWindowRateLimiter


_ALLOWED_OUTPUT_KINDS = {"auto", "dataframe", "chart", "text"}
_STRICT_JSON_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok", "guardrail"]},
        "message": {"type": "string"},
        "code": {"type": "string"},
        "output_kind": {
            "type": "string",
            "enum": ["auto", "dataframe", "chart", "text"],
        },
    },
    "required": ["status", "message", "code", "output_kind"],
    "additionalProperties": False,
}


def response_format_for_model(model: str) -> dict[str, Any]:
    """Use strict schema decoding where Groq currently supports it."""

    if model in _STRICT_JSON_MODELS:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "data_analyst_plan",
                "strict": True,
                "schema": _PLAN_JSON_SCHEMA,
            },
        }
    return {"type": "json_object"}


def model_options(model: str) -> dict[str, Any]:
    """Keep GPT-OSS reasoning private and small enough for free-tier use."""

    if model.startswith("openai/gpt-oss-"):
        return {"include_reasoning": False, "reasoning_effort": "low"}
    return {}


def _provider_error(exc: Exception) -> GatewayError:
    """Map SDK failures without forwarding response bodies, prompts, or keys."""

    status_code = getattr(exc, "status_code", None)
    class_name = type(exc).__name__.lower()
    if status_code == 429 or "ratelimit" in class_name or "rate_limit" in class_name:
        return GatewayRateLimitError(
            "Groq's rate limit was reached. Retry after the provider window resets."
        )
    if (
        status_code in {401, 403}
        or "authentication" in class_name
        or "permission" in class_name
    ):
        return GatewayError("The Groq API key is invalid or lacks model access.")
    if status_code == 404:
        return GatewayError("The configured Groq model is unavailable to this project.")
    if isinstance(status_code, int) and status_code >= 500:
        return GatewayError("Groq is temporarily unavailable.")
    if "timeout" in class_name:
        return GatewayError("The Groq request timed out without a response.")
    if "connection" in class_name:
        return GatewayError("The app could not connect to Groq.")
    return GatewayError("Groq could not complete the analysis request.")


class GroqGateway:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_completion_tokens: int,
        limiter: SlidingWindowRateLimiter | None = None,
    ) -> None:
        if not api_key.strip():
            raise GatewayError("The Groq API key is not configured.")
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.limiter = limiter
        try:
            from groq import Groq
        except ImportError as exc:
            raise GatewayError("The official groq package is not installed.") from exc
        self.client = Groq(api_key=api_key, timeout=timeout_seconds, max_retries=0)

    def generate(self, user_request: str, dataset_context: str) -> ModelPlan:
        return self._complete(generation_user_prompt(user_request, dataset_context))

    def repair(
        self,
        user_request: str,
        dataset_context: str,
        previous_code: str,
        safe_error: str,
    ) -> ModelPlan:
        return self._complete(
            repair_user_prompt(user_request, dataset_context, previous_code, safe_error)
        )

    def _complete(self, user_content: str) -> ModelPlan:
        if self.limiter is not None:
            self.limiter.consume()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
            "max_completion_tokens": self.max_completion_tokens,
            "response_format": response_format_for_model(self.model),
            **model_options(self.model),
        }
        try:
            completion = self.client.chat.completions.create(**request)
        except Exception as exc:  # Provider-specific types change across SDK releases.
            raise _provider_error(exc) from exc
        content = completion.choices[0].message.content or ""
        return parse_model_plan(content)


def _load_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ModelResponseError("The model did not return valid structured JSON.")
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ModelResponseError(
                "The model returned malformed structured output."
            ) from exc
    if not isinstance(parsed, dict):
        raise ModelResponseError("The model response must be one JSON object.")
    return parsed


def parse_model_plan(content: str) -> ModelPlan:
    """Validate and normalize the model's JSON contract."""

    if "GUARDRAIL_TRIGGERED" in content and not content.lstrip().startswith("{"):
        return ModelPlan(
            status="guardrail",
            message=(
                "GUARDRAIL_TRIGGERED: This playground only handles questions "
                "about the uploaded dataset."
            ),
        )

    payload = _load_json_object(content)
    raw_status = str(payload.get("status", "")).strip().lower()
    raw_message = str(payload.get("message", "")).strip()
    raw_code = str(payload.get("code", ""))
    output_kind = str(payload.get("output_kind", "auto")).strip().lower()

    if raw_status not in {"ok", "guardrail"}:
        raise ModelResponseError("The model returned an unsupported status.")
    if output_kind not in _ALLOWED_OUTPUT_KINDS:
        output_kind = "auto"
    if raw_status == "guardrail":
        if "GUARDRAIL_TRIGGERED" not in raw_message:
            raw_message = (
                "GUARDRAIL_TRIGGERED: This playground only handles questions "
                "about the uploaded dataset."
            )
        return ModelPlan(
            status="guardrail",
            message=raw_message,
            code="",
            output_kind="text",
        )
    if not raw_code.strip():
        raise ModelResponseError("The model returned no executable analysis code.")
    return ModelPlan(
        status="ok",
        message=raw_message or "Analysis generated.",
        code=raw_code.strip(),
        output_kind=output_kind,  # type: ignore[arg-type]
    )
