"""Tests for the resilience layer (retry, fallback, circuit breaker).

Fully deterministic and offline: we build a tiny fake registry whose tools fail
on command, then assert the error handler retries transient failures, routes to
fallbacks, and trips the circuit breaker as designed. No LLM, no network — so
this validates the resilience logic regardless of API quota.

Run:  python -m tests.test_error_handling   (or: pytest tests/test_error_handling.py)
"""

from __future__ import annotations

from agent.error_handler import call_with_resilience, CircuitBreaker


# --- A minimal fake registry mirroring the real one's interface --------------

class _FakeTool:
    def __init__(self, name, fn, fallbacks=None):
        self.name = name
        self.fn = fn
        self.fallbacks = fallbacks or []
        self.parameters = {"type": "object", "properties": {}, "required": []}


class _FakeRegistry:
    """Mimics ToolRegistry.call / get / has for testing the handler in isolation."""
    def __init__(self):
        self._tools = {}

    def add(self, name, fn, fallbacks=None):
        self._tools[name] = _FakeTool(name, fn, fallbacks)

    def has(self, name):
        return name in self._tools

    def get(self, name):
        return self._tools.get(name)

    def call(self, name, args):
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"Unknown tool '{name}'."}
        return tool.fn(args)


_NO_SLEEP = lambda _: None  # noqa: E731 - skip real backoff waits in tests


def test_succeeds_first_try():
    reg = _FakeRegistry()
    reg.add("good", lambda a: {"ok": True, "data": 1})
    r = call_with_resilience(reg, "good", {}, _sleep=_NO_SLEEP)
    assert r["ok"] is True
    print("✓ a working tool succeeds immediately")


def test_retries_then_succeeds():
    """A tool that fails twice then succeeds should be retried into success."""
    calls = {"n": 0}

    def flaky(_args):
        calls["n"] += 1
        if calls["n"] < 3:
            return {"ok": False, "error": "temporary network glitch"}
        return {"ok": True, "data": "recovered"}

    reg = _FakeRegistry()
    reg.add("flaky", flaky)
    r = call_with_resilience(reg, "flaky", {}, max_retries=2, _sleep=_NO_SLEEP)
    assert r["ok"] is True, "should recover after retries"
    assert calls["n"] == 3, f"expected 3 attempts, got {calls['n']}"
    print(f"✓ transient failure retried and recovered (after {calls['n']} attempts)")


def test_non_retryable_fails_fast():
    """Validation-type errors must NOT be retried (same call can't improve)."""
    calls = {"n": 0}

    def bad_args(_args):
        calls["n"] += 1
        return {"ok": False, "error": "Invalid arguments for tool: missing 'ticker'."}

    reg = _FakeRegistry()
    reg.add("strict", bad_args)
    r = call_with_resilience(reg, "strict", {}, max_retries=3, _sleep=_NO_SLEEP)
    assert r["ok"] is False
    assert calls["n"] == 1, f"non-retryable error retried {calls['n']} times (should be 1)"
    print("✓ non-retryable (bad-args) error fails fast, no wasted retries")


def test_falls_back_when_primary_fails():
    """A dead primary tool should route to its fallback, which succeeds."""
    reg = _FakeRegistry()
    reg.add("primary", lambda a: {"ok": False, "error": "service unavailable"},
            fallbacks=["backup"])
    reg.add("backup", lambda a: {"ok": True, "data": "from backup"})
    r = call_with_resilience(reg, "primary", {"ticker": "MSFT"}, max_retries=1, _sleep=_NO_SLEEP)
    assert r["ok"] is True, "should succeed via fallback"
    assert r["data"] == "from backup"
    assert any("falling back" in step for step in r.get("_resilience", []))
    print("✓ failed primary routed to working fallback")


def test_circuit_breaker_opens():
    """After repeated failures, the breaker should open for that tool."""
    breaker = CircuitBreaker(failure_threshold=2)

    def always_fails(_args):
        return {"ok": False, "error": "still down"}

    reg = _FakeRegistry()
    reg.add("deadsource", always_fails)  # no fallback
    # First call: fails, retries, records failures -> breaker should trip.
    call_with_resilience(reg, "deadsource", {}, max_retries=2, breaker=breaker, _sleep=_NO_SLEEP)
    assert breaker.is_open("deadsource"), "breaker should be open after repeated failures"
    print(f"✓ circuit breaker opened after failures: {breaker.status()}")


def test_breaker_skips_open_tool_uses_fallback():
    """An already-open tool should skip straight to fallback on next call."""
    breaker = CircuitBreaker(failure_threshold=1)
    calls = {"primary": 0}

    def failing_primary(_args):
        calls["primary"] += 1
        return {"ok": False, "error": "down"}

    reg = _FakeRegistry()
    reg.add("primary", failing_primary, fallbacks=["backup"])
    reg.add("backup", lambda a: {"ok": True, "data": "backup ok"})

    # First call trips the breaker (threshold 1) and uses fallback.
    call_with_resilience(reg, "primary", {}, max_retries=0, breaker=breaker, _sleep=_NO_SLEEP)
    calls_after_first = calls["primary"]

    # Second call: breaker open -> should NOT call primary again, goes to fallback.
    r = call_with_resilience(reg, "primary", {}, max_retries=0, breaker=breaker, _sleep=_NO_SLEEP)
    assert r["ok"] is True
    assert calls["primary"] == calls_after_first, "open breaker should skip the primary call"
    print("✓ open breaker skips dead tool, goes straight to fallback")


def _run_all():
    tests = [
        test_succeeds_first_try,
        test_retries_then_succeeds,
        test_non_retryable_fails_fast,
        test_falls_back_when_primary_fails,
        test_circuit_breaker_opens,
        test_breaker_skips_open_tool_uses_fallback,
    ]
    print(f"Running {len(tests)} resilience tests...\n")
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all