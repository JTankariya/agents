"""Application-specific exceptions with safe, user-displayable messages."""


class AgentPlaygroundError(Exception):
    """Base class for expected application errors."""


class DatasetError(AgentPlaygroundError):
    """Raised when an uploaded dataset cannot be safely loaded."""


class GatewayError(AgentPlaygroundError):
    """Raised when the LLM gateway cannot complete a request."""


class GatewayRateLimitError(GatewayError):
    """Raised when either the local or provider rate limit is reached."""


class ModelResponseError(GatewayError):
    """Raised when the model response does not match the required contract."""


class SandboxError(AgentPlaygroundError):
    """Base class for generated-code sandbox failures."""


class SandboxValidationError(SandboxError):
    """Raised when generated code violates the static sandbox policy."""


class SandboxExecutionError(SandboxError):
    """Raised when generated code fails during isolated execution."""


class SandboxTimeoutError(SandboxError):
    """Raised when generated code exceeds its execution deadline."""


class ContactError(AgentPlaygroundError):
    """Raised when a Formspree submission cannot be completed."""
