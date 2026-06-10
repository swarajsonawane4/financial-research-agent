"""Tests for the evaluation framework.

Deterministic and offline (LLM judge disabled): feeds canned reports + run data
to the metrics and asserts the scorecard behaves — good reports score high, thin
or ungrounded reports score low, and the structure is well-formed. Proves the
20+ metric framework works without API quota.

Run:  python -m tests.test_evaluation   (or: pytest tests/test_evaluation.py)
"""

from __future__ import annotations

from evaluation.metrics import evaluate_report, format_scorecard


_GOOD_REPORT = (
    "# Investment Research Report — MSFT\n\n"
    "## Executive Summary\nMicrosoft delivered strong FY2025 results with broad-based "
    "growth across cloud and productivity segments, though valuation remains elevated.\n\n"
    "## Company Overview\nMicrosoft operates in software and cloud infrastructure.\n\n"
    "## Financial Highlights\nRevenue reached 281,724,000,000 in FY2025, up 18.3%. "
    "Profit margin was 39.3%, return on equity 34.0%, and trailing P/E 25.4. "
    "The balance sheet shows a debt-to-equity ratio of 30.3.\n\n"
    "## Recent Developments\nAzure surpassed 75 billion in annual revenue; heavy AI investment continues.\n\n"
    "## Risks & Considerations\nKey risks include elevated valuation, intense competition "
    "in cloud, and regulatory uncertainty around AI. Debt levels remain manageable.\n\n"
    "## Conclusion\nMicrosoft is well-positioned but priced for growth. This report is "
    "for informational purposes only and does not constitute financial advice.\n\n"
    "## Sources\n- SEC EDGAR 10-K (2025)\n- yfinance\n- Tavily web search"
)

_GOOD_RESULTS = [
    {"tool": "sec_filing_search", "args": {"ticker": "MSFT"},
     "observation": {"ok": True, "url": "https://sec.gov/x", "filing_type": "10-K"}},
    {"tool": "financial_data_api", "args": {"ticker": "MSFT"},
     "observation": {"ok": True, "ticker": "MSFT",
                     "income_statement": {"2025": {"Total Revenue": 281724000000.0}},
                     "ratios": {"profit_margin": 0.393, "return_on_equity": 0.34,
                                "trailing_pe": 25.4, "debt_to_equity": 30.3,
                                "revenue_growth": 0.183}}},
    {"tool": "web_search", "args": {"query": "MSFT"},
     "observation": {"ok": True, "results": [{"title": "Azure news", "url": "x"}]}},
]


def test_good_report_scores_well():
    ev = evaluate_report(_GOOD_REPORT, results=_GOOD_RESULTS, plan=[1, 2, 3],
                         sources=["SEC EDGAR 10-K", "yfinance", "Tavily"],
                         use_llm_judge=False)
    assert ev["overall"] >= 0.7, f"good report should score >=0.7, got {ev['overall']}"
    assert ev["by_category"]["Accuracy"]["score"] >= 0.7
    print(f"✓ good report scores well: overall {ev['overall']:.2f}")


def test_has_20_plus_metrics():
    """The framework must define 20+ metrics (incl. LLM-judged + registry)."""
    class _FakeReg:
        calls_total = 5
        calls_useful = 4
        memory_hits = 1
        external_api_calls = 4
        def tool_efficiency(self): return 0.8
        def memory_utilization(self): return 0.25
    ev = evaluate_report(_GOOD_REPORT, results=_GOOD_RESULTS, plan=[1, 2, 3],
                         sources=["x"], registry=_FakeReg(), use_llm_judge=True)
    assert ev["total"] >= 20, f"expected 20+ metrics, got {ev['total']}"
    print(f"✓ framework defines {ev['total']} metrics (>=20 required)")


def test_thin_report_scores_lower_on_depth():
    thin = "MSFT is good. Revenue 281,724,000,000."
    ev = evaluate_report(thin, results=_GOOD_RESULTS, plan=[1], use_llm_judge=False)
    depth = next(m for m in ev["metrics"] if m.code == "CM-3")
    assert depth.score < 0.5, "a 5-word report should score low on depth"
    print(f"✓ thin report flagged: depth {depth.score:.2f}")


def test_ungrounded_numbers_flagged():
    """A report full of numbers NOT in the gathered data should score low on AC-1."""
    fabricated = ("Revenue was 999,888,777,000 with a 88.8% margin and P/E of 12.3. "
                  "This is not financial advice.")
    ev = evaluate_report(fabricated, results=_GOOD_RESULTS, plan=[1], use_llm_judge=False)
    grounding = next(m for m in ev["metrics"] if m.code == "AC-1")
    assert grounding.score is not None and grounding.score < 0.6, \
        f"fabricated numbers should score low on grounding, got {grounding.score}"
    print(f"✓ ungrounded numbers flagged: grounding {grounding.score:.2f}")


def test_missing_disclaimer_flagged():
    no_disclaimer = "# Report\n## Conclusion\nBuy this stock now, guaranteed returns."
    ev = evaluate_report(no_disclaimer, results=_GOOD_RESULTS, plan=[1], use_llm_judge=False)
    disc = next(m for m in ev["metrics"] if m.code == "AC-3")
    assert disc.score == 0.0, "missing disclaimer should score 0"
    print("✓ missing disclaimer flagged")


def test_scorecard_renders():
    ev = evaluate_report(_GOOD_REPORT, results=_GOOD_RESULTS, plan=[1, 2, 3],
                         sources=["x"], use_llm_judge=False)
    text = format_scorecard(ev)
    assert "EVALUATION SCORECARD" in text and "Accuracy" in text
    print("✓ scorecard renders as readable text")


def _run_all():
    tests = [
        test_good_report_scores_well,
        test_has_20_plus_metrics,
        test_thin_report_scores_lower_on_depth,
        test_ungrounded_numbers_flagged,
        test_missing_disclaimer_flagged,
        test_scorecard_renders,
    ]
    print(f"Running {len(tests)} evaluation-framework tests...\n")
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()