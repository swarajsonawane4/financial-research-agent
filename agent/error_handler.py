"""Error handling — retry, fallback chains, and a circuit breaker.

The registry's `call()` validates and dispatches a single tool, returning a
clean {"ok": False, "error": ...} on failure rather than crashing. This module
adds the *active* resilience layer on top of that:

  1. RETRY WITH BACKOFF — transient failures (network blips, rate limits) are
     retried a few times with increasing waits, since many failures are
     temporary. Validation errors (bad args) are NOT retried — retrying the same
     bad call can't help, so we fail fast on those.

  2. FALLBACK CHAINS — if a tool keeps failing, automatically try a different
     tool that can do a similar job (using the `fallbacks` list on each Tool).
     e.g. financial_data_api -> company_profile for at least basic info.

  3. CIRCUIT BREAKER — if a tool fails repeatedly within a run, stop calling it
     (mark it "open") so we don't waste time/quota hammering a dead source.
     Subsequent calls to an open tool short-circuit straight to its fallback.

This is what Challenge 8 (the "50% tool failure" stress test) exercises: the
agent must keep producing a useful report even when tools are failing.

Design note: this wraps the registry rather than living inside it, keeping the
registry's job (validate + dispatch one tool) separate from the resilience
policy (how hard to try, what to try instead). Separation of concerns.
"""

from __future__ import annotations

import time
from typing import Optional


# Errors containing these markers are NOT worth retrying (the call is malformed,
# so the same call will fail identically). Everything else is treated as
# potentially transient and eligible for retry.
_NON_RETRYABLE_MARKERS = ("Invalid arguments", "Unknown tool", "Unsupported")


class CircuitBreaker:
    """Tracks per-tool failures within a run and 'opens' on repeated failure.

    Once a tool's failure count hits the threshold, the circuit is 'open' and
    further calls to that tool are skipped (routed to fallback instead), until
    the run ends. Simple per-run breaker — no time-based half-open state needed
    for a single research run.
    """

    def __init__(self, failure_threshold: int = 2) -> None:
        self.failure_threshold = failure_threshold
        self._failures: dict[str, int] = {}

    def record_failure(self, tool: str) -> None:
        self._failures[tool] = self._failures.get(tool, 0) + 1

    def record_success(self, tool: str) -> None:
        # A success resets the count — the tool is evidently working again.
        self._failures[tool] = 0

    def is_open(self, tool: str) -> bool:
        """True if this tool has failed enough to be considered down."""
        return self._failures.get(tool, 0) >= self.failure_threshold

    def status(self) -> dict:
        return {t: ("open" if c >= self.failure_threshold else "closed")
                for t, c in self._failures.items() if c > 0}


def _is_retryable(error: str) -> bool:
    """Whether an error string represents a transient (retryable) failure."""
    return not any(marker in error for marker in _NON_RETRYABLE_MARKERS)


def call_with_resilience(
    registry,
    name: str,
    args: dict,
    *,
    breaker: Optional[CircuitBreaker] = None,
    max_retries: int = 2,
    base_delay: float = 0.5,
    _depth: int = 0,
    _sleep=time.sleep,
) -> dict:
    """Execute a tool call with retry, circuit-breaking, and fallback.

    Args:
        registry: the ToolRegistry instance.
        name: tool to call.
        args: arguments for the tool.
        breaker: a CircuitBreaker (shared across a run); created if None.
        max_retries: retry attempts for transient failures (per tool).
        base_delay: initial backoff delay; doubles each retry (exponential).
        _depth: internal recursion guard for fallback chains.
        _sleep: injectable sleep (tests pass a no-op to avoid real waits).

    Returns the tool result dict. On total failure (tool + all fallbacks
    exhausted), returns the last error dict with a `_resilience` trace.
    """
    breaker = breaker or CircuitBreaker()
    trace = []

    # If the breaker has already tripped for this tool, skip straight to fallback.
    if breaker.is_open(name):
        trace.append(f"{name}: circuit open, skipping")
        return _try_fallbacks(registry, name, args, breaker, max_retries,
                              base_delay, _depth, _sleep, trace,
                              last_error="circuit open")

    # Attempt the call, retrying transient failures with exponential backoff.
    last_error = ""
    attempts = max_retries + 1
    for attempt in range(attempts):
        result = registry.call(name, args)
        if result.get("ok"):
            breaker.record_success(name)
            if trace:
                result.setdefault("_resilience", trace + [f"{name}: ok"])
            return result

        last_error = result.get("error", "unknown error")

        # Don't retry malformed calls — they'll fail identically every time.
        if not _is_retryable(last_error):
            trace.append(f"{name}: non-retryable ({last_error})")
            breaker.record_failure(name)
            break

        trace.append(f"{name}: attempt {attempt + 1} failed ({last_error})")
        breaker.record_failure(name)

        # Back off before the next attempt (skip the wait after the last try).
        if attempt < attempts - 1:
            _sleep(base_delay * (2 ** attempt))

    # Tool exhausted its retries (or hit a non-retryable error) — try fallbacks.
    return _try_fallbacks(registry, name, args, breaker, max_retries,
                          base_delay, _depth, _sleep, trace, last_error)


def _try_fallbacks(registry, name, args, breaker, max_retries, base_delay,
                   _depth, _sleep, trace, last_error) -> dict:
    """Try each fallback tool for `name` in order, until one succeeds."""
    tool = registry.get(name)
    fallbacks = tool.fallbacks if tool else []

    # Guard against runaway fallback recursion.
    if _depth >= 3 or not fallbacks:
        return {
            "ok": False,
            "error": f"{name} failed and no fallback succeeded. Last error: {last_error}",
            "_resilience": trace,
        }

    for fb in fallbacks:
        if not registry.has(fb):
            trace.append(f"fallback {fb}: not registered, skipping")
            continue
        trace.append(f"falling back: {name} -> {fb}")
        # Fallback args: keep shared keys the fallback understands (ticker, query).
        fb_args = _adapt_args(args)
        result = call_with_resilience(
            registry, fb, fb_args, breaker=breaker, max_retries=max_retries,
            base_delay=base_delay, _depth=_depth + 1, _sleep=_sleep,
        )
        prior = result.get("_resilience", [])
        result["_resilience"] = trace + prior
        if result.get("ok"):
            return result

    return {
        "ok": False,
        "error": f"{name} and all fallbacks failed. Last error: {last_error}",
        "_resilience": trace,
    }


def _adapt_args(args: dict) -> dict:
    """Pass through args a fallback tool is likely to share (ticker/query)."""
    shared = {}
    for key in ("ticker", "query"):
        if key in args:
            shared[key] = args[key]
    # If no shared keys, pass the original args and let validation sort it out.
    return shared or dict(args)