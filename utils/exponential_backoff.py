"""Retry temporary API failures with bounded exponential backoff and jitter."""

import random
import time
from functools import wraps


def error_kind(error):
    code = getattr(error, "status_code", None) or getattr(error, "code", None)
    if code is None:
        code = getattr(getattr(error, "response", None), "status_code", None)
    code = str(code).upper()
    if code in {
        "400",
        "401",
        "403",
        "404",
        "422",
        "INVALID_ARGUMENT",
        "UNAUTHENTICATED",
        "PERMISSION_DENIED",
    }:
        return "invalid_request"
    if isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower():
        return "timeout"
    if (
        code
        in {
            "408",
            "429",
            "500",
            "502",
            "503",
            "504",
            "RESOURCE_EXHAUSTED",
            "UNAVAILABLE",
            "DEADLINE_EXCEEDED",
        }
        or isinstance(error, ConnectionError)
        or "connection" in type(error).__name__.lower()
    ):
        return "transient_api"
    if isinstance(error, OSError):
        return "infrastructure"
    return "api_error"


def retry_with_exponential_backoff(
    func, max_attempts=10, initial_sleep_time=1.0, backoff_factor=2.0
):
    @wraps(func)
    def wrapper(*args, **kwargs):
        attempts = max_attempts
        if attempts < 1:
            raise ValueError("max_attempts must be at least one")
        for attempt in range(attempts):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if (
                    error_kind(exc) not in {"timeout", "transient_api"}
                    or attempt == attempts - 1
                ):
                    raise
                delay = min(60.0, initial_sleep_time * backoff_factor**attempt)
                time.sleep(random.uniform(delay / 2, delay))

    return wrapper
