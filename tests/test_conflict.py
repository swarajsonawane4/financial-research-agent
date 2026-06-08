"""Tests for the conflict-resolution engine.

These are DETERMINISTIC: they feed the engine known, hand-crafted conflicts and
assert it resolves each one correctly by source tier. This is the reliable way
to validate conflict resolution — live runs depend on the LLM being available
AND the day's news happening to contradict the fundamentals, neither of which we
control. The resolution logic itself (`_resolve_one`) is pure tier-based code
and needs no LLM or network, so it can be tested directly and repeatably.

Run:  python -m tests.test_conflict      (or: pytest tests/test_conflict.py)
"""

from __future__ import annotations

import sys
import types

# The detection step imports google.generativeai; stub it so these tests run
# even where that package isn't installed (we test resolution, not detection).
if "google.generativeai" not in sys.modules:
    try:
        import google.generativeai  # noqa: F401
    except ImportError:
        sys.modules["google.generativeai"] = types.ModuleType("google.generativeai")

from synthesis.conflict import _resolve_one, _tool_tier, conflicts_for_prompt


def test_tier_mapping():
    """Source tiers should rank filings above financials above news."""
    assert _tool_tier("sec_filing_search") == 1
    assert _tool_tier("financial_data_api") == 2
    assert _tool_tier("web_search") == 3
    assert _tool_tier("news_sentiment") == 3
    assert _tool_tier("earnings_transcript") == 4
    # higher number = less reliable; filing must outrank news
    assert _tool_tier("sec_filing_search") < _tool_tier("news_sentiment")
    print("✓ tier mapping correct (filing < financials < news)")


def test_numeric_conflict_trusts_highest_tier():
    """A filing vs. news numeric disagreement should resolve to the filing."""
    conflict = {
        "topic": "FY2025 revenue",
        "type": "numeric",
        "claims": [
            {"tool": "sec_filing_search", "value": "$2.1B"},
            {"tool": "web_search", "value": "$2.3B"},
        ],
    }
    r = _resolve_one(conflict)
    assert r["unresolved"] is False, "clear-tier numeric conflict should resolve"
    assert r["resolved_value"] == "$2.1B", "should trust the SEC filing's figure"
    print(f"✓ numeric conflict resolved to filing: {r['resolved_value']}")


def test_numeric_tie_is_unresolved():
    """Two equally-authoritative numeric sources disagreeing = unresolved."""
    conflict = {
        "topic": "revenue",
        "type": "numeric",
        "claims": [
            {"tool": "financial_data_api", "value": "$2.1B"},   # tier 2
            {"tool": "peer_comparison", "value": "$2.4B"},      # tier 2
        ],
    }
    r = _resolve_one(conflict)
    assert r["unresolved"] is True, "equal-tier numeric conflict cannot be adjudicated"
    assert r["resolved_value"] is None
    print("✓ equal-tier numeric conflict correctly left unresolved")


def test_qualitative_conflict_is_surfaced_not_forced():
    """Strong fundamentals vs. negative sentiment: surface tension, don't pick."""
    conflict = {
        "topic": "company health",
        "type": "qualitative",
        "claims": [
            {"tool": "financial_data_api", "value": "strong: 43% margin, 84% growth"},
            {"tool": "news_sentiment", "value": "negative market sentiment"},
        ],
    }
    r = _resolve_one(conflict)
    assert r["unresolved"] is True, "qualitative tension should not be force-resolved"
    assert r["resolved_value"] is None
    assert "valid simultaneously" in r["resolution"]
    print("✓ qualitative conflict surfaced (not forced to a winner)")


def test_prompt_rendering_includes_resolution():
    """The text fed to synthesis should state the trusted value."""
    conflict = {
        "topic": "revenue", "type": "numeric",
        "claims": [
            {"tool": "sec_filing_search", "value": "$2.1B"},
            {"tool": "web_search", "value": "$2.3B"},
        ],
    }
    rendered = conflicts_for_prompt([_resolve_one(conflict)])
    assert "Trusted value: $2.1B" in rendered
    assert "sec_filing_search" in rendered
    print("✓ prompt rendering surfaces the resolution for synthesis")


def _run_all():
    tests = [
        test_tier_mapping,
        test_numeric_conflict_trusts_highest_tier,
        test_numeric_tie_is_unresolved,
        test_qualitative_conflict_is_surfaced_not_forced,
        test_prompt_rendering_includes_resolution,
    ]
    print(f"Running {len(tests)} conflict-resolution tests...\n")
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()