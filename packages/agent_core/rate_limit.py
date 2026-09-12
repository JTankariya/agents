"""Small in-process sliding-window request limiter."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

from .errors import GatewayRateLimitError


class SlidingWindowRateLimiter:
    """Cap calls in one app process; this is not identity-based abuse control."""

    def __init__(
        self,
        *,
        max_requests: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clock = clock
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def consume(self) -> None:
        now = self.clock()
        cutoff = now - self.window_seconds
        with self._lock:
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.max_requests:
                raise GatewayRateLimitError(
                    "The shared demo request limit was reached. Try again after the window resets."
                )
            self._timestamps.append(now)
