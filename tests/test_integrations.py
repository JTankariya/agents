from types import SimpleNamespace

import pytest

from packages.agent_core import contact
from packages.agent_core.errors import ContactError, GatewayError, GatewayRateLimitError
from packages.agent_core.groq_gateway import _provider_error


class _ProviderFailure(Exception):
    def __init__(self, status_code: int | None = None) -> None:
        super().__init__("sensitive provider response must not be forwarded")
        self.status_code = status_code


class APITimeoutError(Exception):
    pass


def test_provider_rate_limit_is_structured_and_redacted():
    mapped = _provider_error(_ProviderFailure(429))
    assert isinstance(mapped, GatewayRateLimitError)
    assert "sensitive provider response" not in str(mapped)


@pytest.mark.parametrize("status_code", [401, 403])
def test_provider_auth_failures_are_structured(status_code: int):
    mapped = _provider_error(_ProviderFailure(status_code))
    assert isinstance(mapped, GatewayError)
    assert "invalid" in str(mapped).lower()


def test_provider_missing_model_is_structured():
    mapped = _provider_error(_ProviderFailure(404))
    assert "model" in str(mapped).lower()
    assert "unavailable" in str(mapped).lower()


def test_provider_server_failure_is_structured():
    mapped = _provider_error(_ProviderFailure(503))
    assert str(mapped) == "Groq is temporarily unavailable."


def test_provider_timeout_is_structured():
    mapped = _provider_error(APITimeoutError("private request details"))
    assert "timed out" in str(mapped).lower()
    assert "private request details" not in str(mapped)


def test_formspree_rate_limit_is_structured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        contact.requests,
        "post",
        lambda *args, **kwargs: SimpleNamespace(status_code=429),
    )
    with pytest.raises(ContactError, match="rate limit"):
        contact.submit_formspree(form_id="abcd1234", fields={"message": "hello"})
