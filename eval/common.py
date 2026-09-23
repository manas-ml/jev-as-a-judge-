"""Shared utilities: env loading, retry/backoff, JSONL audit logging, cost/quality mapping.

Not part of the spec's file list explicitly, but factored out so run_agent.py,
repeatability.py, and the judge don't duplicate the same retry/logging boilerplate.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "results" / "calls.jsonl"

_ENV_LOADED = False


def load_env() -> None:
    """Load .env from repo root exactly once."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    load_dotenv(ROOT / ".env")
    _ENV_LOADED = True


def require_env(name: str) -> str:
    load_env()
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name!r}. Copy .env.example to .env and fill it in."
        )
    return value


# --- Fixed mapping from Jev's 'search_behavior' choice to a 0-1 number ---
# Documented per the spec: used to fold the categorical search_behavior answer
# into the single composite `quality` score.
SEARCH_BEHAVIOR_SCORE = {
    "appropriate": 1.0,
    "unnecessary": 0.5,
    "failed": 0.0,
}

# quality >= this threshold => does_pass = 1 (documented threshold, see README).
DOES_PASS_THRESHOLD = 0.5


def compute_quality(groundedness: float, search_behavior: str, usefulness: float) -> float:
    mapped = SEARCH_BEHAVIOR_SCORE.get(search_behavior, 0.0)
    return (float(groundedness) + mapped + float(usefulness)) / 3.0


def derive_does_pass(quality: float) -> int:
    return 1 if quality >= DOES_PASS_THRESHOLD else 0


# --- Append-only JSONL audit log for every judge/agent API call ---


def log_call(record: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.time(), **record}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# --- Retry with exponential backoff on 429/5xx ---

T = TypeVar("T")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _status_code_of(exc: BaseException) -> int | None:
    for attr in ("status_code",):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    response = getattr(exc, "response", None)
    if response is not None:
        val = getattr(response, "status_code", None)
        if isinstance(val, int):
            return val
    return None


def is_retryable(exc: BaseException) -> bool:
    code = _status_code_of(exc)
    if code is not None:
        return code in RETRYABLE_STATUS_CODES
    # Connection-level failures (timeouts, resets) are also worth retrying.
    name = type(exc).__name__
    return any(k in name for k in ("Timeout", "Connection", "ConnectError", "RemoteProtocolError"))


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Call fn(), retrying on retryable errors with exponential backoff + jitter."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad, re-raised when not retryable
            if attempt >= max_attempts or not is_retryable(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = delay * (0.5 + random.random())  # jitter
            if on_retry:
                on_retry(attempt, exc, delay)
            time.sleep(delay)
