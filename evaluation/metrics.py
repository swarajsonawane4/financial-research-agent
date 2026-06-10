"""Evaluation framework — 20+ quality metrics across 5 categories.

Turns "the agent seems to work" into measurable scores. Following the reference
spec's five metric categories (Section A5.2):

  1. ACCURACY      — are the reported numbers/claims correct and grounded?
  2. COMPLETENESS  — did the report cover what a real analyst report should?
  3. SOURCE QUALITY— are sources cited, reliable, and diverse?
  4. AGENT BEHAVIOR— efficiency of tool use, memory utilization (the AB-series;
                     AB-1 and AB-4 are computed live in the registry).
  5. REPORT QUALITY— coherence, readability, structure (LLM-as-judge).

Design: the bulk of these are DETERMINISTIC — they inspect the report text and
the run's structured data, no LLM needed. That makes evaluation cheap, fast, and
repeatable (it runs without API quota). A small number of qualitative metrics
(coherence, depth) use an LLM judge when available, and degrade to "not scored"
when it isn't, rather than blocking the whole evaluation.

Each metric returns a 0..1 score (or None if not applicable), plus a short note.
`evaluate_report()` runs them all and returns a category-grouped scorecard.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None


@dataclass
class MetricResult:
    code: str
    name: str
    category: str
    score: Optional[float]   # 0..1, or None if not applicable / not scored
    note: str = ""


# ============================================================================
# CATEGORY 1 — ACCURACY
# ============================================================================

def numeric_grounding(report_text: str, gathered_numbers: list) -> MetricResult:
    """AC-1: fraction of numbers in the report that trace to gathered data.

    Extracts large numbers / percentages from the report and checks how many
    appear in (or are derivable from) the data the agent actually gathered. Low
    scores suggest fabricated figures.
    """
    report_nums = _extract_numbers(report_text)
    if not report_nums:
        return MetricResult("AC-1", "Numeric grounding", "Accuracy", None,
                            "No numbers in report to check.")
    gathered_set = set()
    for n in gathered_numbers:
        gathered_set.add(round(n, 2))
        gathered_set.add(round(n))  # also the rounded form
    grounded = 0
    for n in report_nums:
        # A report number is "grounded" if it (or a rounded form) was gathered.
        if round(n, 2) in gathered_set or round(n) in gathered_set or _is_derivable(n, gathered_numbers):
            grounded += 1
    score = grounded / len(report_nums)
    return MetricResult("AC-1", "Numeric grounding", "Accuracy", round(score, 2),
                        f"{grounded}/{len(report_nums)} report numbers trace to gathered data.")


def no_contradiction_with_sources(report_text: str, conflicts: list) -> MetricResult:
    """AC-2: did the report respect resolved conflicts (use trusted values)?"""
    if not conflicts:
        return MetricResult("AC-2", "Conflict adherence", "Accuracy", 1.0,
                            "No conflicts to adhere to.")
    resolved = [c for c in conflicts if c.get("resolved_value")]
    if not resolved:
        return MetricResult("AC-2", "Conflict adherence", "Accuracy", None,
                            "Conflicts present but none had a single resolved value.")
    honored = sum(1 for c in resolved if str(c["resolved_value"]) in report_text)
    score = honored / len(resolved)
    return MetricResult("AC-2", "Conflict adherence", "Accuracy", round(score, 2),
                        f"{honored}/{len(resolved)} resolved values appear in the report.")


def disclaimer_present(report_text: str) -> MetricResult:
    """AC-3: does the report carry an analysis-not-advice disclaimer?"""
    has = bool(re.search(r"not (constitute |financial )?advice|informational purposes", report_text, re.I))
    return MetricResult("AC-3", "Disclaimer present", "Accuracy", 1.0 if has else 0.0,
                        "Disclaimer found." if has else "No not-financial-advice disclaimer.")


def figure_specificity(report_text: str) -> MetricResult:
    """AC-4: does the report use specific figures rather than vague hand-waving?

    Rewards concrete numbers over phrases like 'significant growth' with no value.
    """
    nums = _extract_numbers(report_text)
    vague = len(re.findall(r"\b(significant|substantial|strong|robust|considerable)\b", report_text, re.I))
    if not nums and not vague:
        return MetricResult("AC-4", "Figure specificity", "Accuracy", None, "Neither figures nor vague terms.")
    # More concrete numbers relative to vague adjectives = better.
    score = min(len(nums) / (len(nums) + vague + 1) + 0.3, 1.0) if (nums or vague) else 0.0
    return MetricResult("AC-4", "Figure specificity", "Accuracy", round(score, 2),
                        f"{len(nums)} concrete figures vs {vague} vague qualifier(s).")


# ============================================================================
# CATEGORY 2 — COMPLETENESS
# ============================================================================

_EXPECTED_SECTIONS = ["executive summary", "financial", "risk", "conclusion"]


def section_coverage(report_text: str) -> MetricResult:
    """CM-1: fraction of expected report sections that are present."""
    low = report_text.lower()
    present = sum(1 for s in _EXPECTED_SECTIONS if s in low)
    score = present / len(_EXPECTED_SECTIONS)
    return MetricResult("CM-1", "Section coverage", "Completeness", round(score, 2),
                        f"{present}/{len(_EXPECTED_SECTIONS)} expected sections present.")


def data_dimension_coverage(results: list) -> MetricResult:
    """CM-2: how many distinct data dimensions the research touched.

    Rewards drawing on multiple kinds of source (filing, financials, news,
    profile, etc.) rather than a single one.
    """
    kinds = set()
    for r in results:
        if r.get("observation", {}).get("ok"):
            kinds.add(_dimension_of(r.get("tool", "")))
    kinds.discard(None)
    # 4+ distinct dimensions = full marks.
    score = min(len(kinds) / 4.0, 1.0)
    return MetricResult("CM-2", "Data dimension coverage", "Completeness", round(score, 2),
                        f"Touched {len(kinds)} distinct data dimension(s): {sorted(kinds)}.")


def report_depth(report_text: str) -> MetricResult:
    """CM-3: a length-based proxy for depth (very short reports score lower)."""
    words = len(report_text.split())
    score = min(words / 300.0, 1.0)  # ~300+ words = full marks
    return MetricResult("CM-3", "Report depth", "Completeness", round(score, 2),
                        f"Report is {words} words.")


def has_structure(report_text: str) -> MetricResult:
    """CM-4: does the report use headings to organize content?"""
    headings = len(re.findall(r"^#{1,3}\s", report_text, re.M))
    score = min(headings / 4.0, 1.0)  # 4+ headings = well-structured
    return MetricResult("CM-4", "Structural organization", "Completeness", round(score, 2),
                        f"{headings} heading(s) found.")


def risk_coverage(report_text: str) -> MetricResult:
    """CM-5: does the report actually discuss risks (not just have the header)?"""
    low = report_text.lower()
    risk_terms = len(re.findall(r"\b(risk|valuation|competition|headwind|concern|uncertainty|debt)\b", low))
    score = min(risk_terms / 3.0, 1.0)
    return MetricResult("CM-5", "Risk coverage", "Completeness", round(score, 2),
                        f"{risk_terms} risk-related mention(s).")


# ============================================================================
# CATEGORY 3 — SOURCE QUALITY
# ============================================================================

def citation_presence(report_text: str, sources: list) -> MetricResult:
    """SQ-1: does the report include a sources/citations section?"""
    has_section = bool(re.search(r"sources?|references?|citations?", report_text, re.I))
    score = 1.0 if (has_section and sources) else (0.5 if sources else 0.0)
    return MetricResult("SQ-1", "Citation presence", "Source quality", score,
                        f"{len(sources)} source(s); citations section {'found' if has_section else 'absent'}.")


def source_reliability(results: list) -> MetricResult:
    """SQ-2: average reliability tier of the sources used (higher = better)."""
    from synthesis.conflict import _tool_tier
    tiers = [_tool_tier(r["tool"]) for r in results if r.get("observation", {}).get("ok")]
    tiers = [t for t in tiers if t < 99]
    if not tiers:
        return MetricResult("SQ-2", "Source reliability", "Source quality", None,
                            "No tiered sources used.")
    avg_tier = sum(tiers) / len(tiers)
    # Map tier 1->1.0 ... tier 5->0.2 (lower tier number = more reliable = higher score).
    score = max(0.0, min(1.0, (6 - avg_tier) / 5))
    return MetricResult("SQ-2", "Source reliability", "Source quality", round(score, 2),
                        f"Average source tier {avg_tier:.1f} (1=best).")


def source_diversity(results: list) -> MetricResult:
    """SQ-3: number of distinct source TYPES, normalized."""
    types = set()
    for r in results:
        if r.get("observation", {}).get("ok"):
            types.add(_dimension_of(r.get("tool", "")))
    types.discard(None)
    score = min(len(types) / 3.0, 1.0)  # 3+ types = full marks
    return MetricResult("SQ-3", "Source diversity", "Source quality", round(score, 2),
                        f"{len(types)} distinct source type(s).")


# ============================================================================
# CATEGORY 4 — AGENT BEHAVIOR (AB-series; AB-1/AB-4 from the registry)
# ============================================================================

def tool_efficiency(registry) -> MetricResult:
    """AB-1: useful tool calls / total tool calls (target >= 0.70)."""
    score = registry.tool_efficiency()
    return MetricResult("AB-1", "Tool efficiency", "Agent behavior", round(score, 2),
                        f"{registry.calls_useful}/{registry.calls_total} calls were useful.")


def memory_utilization(registry) -> MetricResult:
    """AB-4: memory hits / external API calls (target >= 0.30)."""
    score = registry.memory_utilization()
    return MetricResult("AB-4", "Memory utilization", "Agent behavior", round(min(score, 1.0), 2),
                        f"{registry.memory_hits} memory hit(s) vs {registry.external_api_calls} external call(s).")


def plan_executed(results: list, plan: list) -> MetricResult:
    """AB-2: fraction of planned steps that executed successfully."""
    if not plan:
        return MetricResult("AB-2", "Plan execution", "Agent behavior", None, "No plan recorded.")
    ok = sum(1 for r in results if r.get("observation", {}).get("ok"))
    score = ok / len(plan)
    return MetricResult("AB-2", "Plan execution", "Agent behavior", round(score, 2),
                        f"{ok}/{len(plan)} planned steps succeeded.")


def graceful_failure(results: list) -> MetricResult:
    """AB-3: did failures degrade gracefully (no crash, errors reported)?

    If any step failed, full marks require that the run still produced results
    for the others (i.e. failures didn't abort the run).
    """
    total = len(results)
    if total == 0:
        return MetricResult("AB-3", "Graceful degradation", "Agent behavior", None, "No steps.")
    failed = [r for r in results if not r.get("observation", {}).get("ok")]
    if not failed:
        return MetricResult("AB-3", "Graceful degradation", "Agent behavior", 1.0,
                            "No failures occurred.")
    # Graceful = at least one other step still succeeded AND failures carry an error msg.
    succeeded = total - len(failed)
    all_have_errors = all(r["observation"].get("error") for r in failed)
    score = 1.0 if (succeeded > 0 and all_have_errors) else 0.5
    return MetricResult("AB-3", "Graceful degradation", "Agent behavior", score,
                        f"{len(failed)} failed, {succeeded} succeeded; failures reported cleanly.")


def resilience_recovery(results: list) -> MetricResult:
    """AB-5: did the resilience layer successfully recover any failed calls?

    Inspects the _resilience trace for fallbacks that ended in success — evidence
    the agent routed around failures rather than just failing.
    """
    recovered = 0
    for r in results:
        trace = r.get("observation", {}).get("_resilience", [])
        if any("falling back" in t for t in trace) and r["observation"].get("ok"):
            recovered += 1
    # This is informational: 1.0 if any recovery happened, else "not applicable"
    # (most clean runs need no recovery, which shouldn't be penalised).
    if not any(r.get("observation", {}).get("_resilience") for r in results):
        return MetricResult("AB-5", "Resilience recovery", "Agent behavior", None,
                            "No resilience events this run.")
    return MetricResult("AB-5", "Resilience recovery", "Agent behavior", 1.0 if recovered else 0.5,
                        f"{recovered} call(s) recovered via fallback.")


# ============================================================================
# CATEGORY 5 — REPORT QUALITY (LLM-as-judge; degrades when unavailable)
# ============================================================================

_JUDGE_DIMENSIONS = {
    "RQ-1": ("Coherence", "Does the report read as a coherent, logically-ordered analysis?"),
    "RQ-2": ("Clarity", "Is the writing clear and professional, free of jargon-noise?"),
    "RQ-3": ("Analytical depth", "Does it interpret the data, not just list it?"),
    "RQ-4": ("Balance", "Does it present a balanced view (strengths and risks)?"),
}


def report_quality_llm(report_text: str) -> list:
    """RQ-1..4: LLM-judged qualitative scores. Returns 'not scored' if no LLM."""
    if not (genai and os.getenv("GEMINI_API_KEY")):
        return [MetricResult(code, name, "Report quality", None, "LLM unavailable — not scored.")
                for code, (name, _) in _JUDGE_DIMENSIONS.items()]
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        dims = "\n".join(f"- {code} ({name}): {q}" for code, (name, q) in _JUDGE_DIMENSIONS.items())
        prompt = (
            "You are evaluating an investment research report. Score each dimension "
            "from 0.0 to 1.0. Respond with ONLY JSON mapping each code to a score:\n"
            f'{{"RQ-1": 0.0, "RQ-2": 0.0, "RQ-3": 0.0, "RQ-4": 0.0}}\n\n'
            f"DIMENSIONS:\n{dims}\n\nREPORT:\n{report_text[:4000]}\n\nJSON:"
        )
        model = genai.GenerativeModel(DEFAULT_MODEL)
        text = (model.generate_content(prompt).text or "").strip()
        if text.startswith("```"):
            text = text.split("```")[1].replace("json", "", 1).strip().strip("`").strip()
        scores = json.loads(text)
        out = []
        for code, (name, _) in _JUDGE_DIMENSIONS.items():
            s = scores.get(code)
            out.append(MetricResult(code, name, "Report quality",
                                    round(float(s), 2) if s is not None else None,
                                    "LLM judge." if s is not None else "Missing from judge output."))
        return out
    except Exception as exc:  # noqa: BLE001
        return [MetricResult(code, name, "Report quality", None, f"Judge error: {type(exc).__name__}")
                for code, (name, _) in _JUDGE_DIMENSIONS.items()]


# ============================================================================
# Orchestrator
# ============================================================================

def evaluate_report(
    report_text: str,
    *,
    results: Optional[list] = None,
    plan: Optional[list] = None,
    conflicts: Optional[list] = None,
    sources: Optional[list] = None,
    registry=None,
    gathered_numbers: Optional[list] = None,
    use_llm_judge: bool = True,
) -> dict:
    """Run all metrics and return a category-grouped scorecard.

    Returns:
        {
          "metrics": [MetricResult, ...],
          "by_category": {category: {"score": float|None, "metrics": [...]}},
          "overall": float,         # mean of all scored metrics
          "scored": int, "total": int,
        }
    """
    results = results or []
    plan = plan or []
    conflicts = conflicts or []
    sources = sources or []
    gathered_numbers = gathered_numbers or _harvest_numbers(results)

    metrics: list = []
    # Accuracy
    metrics.append(numeric_grounding(report_text, gathered_numbers))
    metrics.append(no_contradiction_with_sources(report_text, conflicts))
    metrics.append(disclaimer_present(report_text))
    metrics.append(figure_specificity(report_text))
    # Completeness
    metrics.append(section_coverage(report_text))
    metrics.append(data_dimension_coverage(results))
    metrics.append(report_depth(report_text))
    metrics.append(has_structure(report_text))
    metrics.append(risk_coverage(report_text))
    # Source quality
    metrics.append(citation_presence(report_text, sources))
    metrics.append(source_reliability(results))
    metrics.append(source_diversity(results))
    # Agent behavior
    if registry is not None:
        metrics.append(tool_efficiency(registry))
        metrics.append(memory_utilization(registry))
    metrics.append(plan_executed(results, plan))
    metrics.append(graceful_failure(results))
    metrics.append(resilience_recovery(results))
    # Report quality (LLM)
    if use_llm_judge:
        metrics.extend(report_quality_llm(report_text))
    else:
        metrics.extend(MetricResult(c, n, "Report quality", None, "Judge disabled.")
                       for c, (n, _) in _JUDGE_DIMENSIONS.items())

    # Group + aggregate.
    by_category: dict = {}
    for m in metrics:
        cat = by_category.setdefault(m.category, {"metrics": [], "score": None})
        cat["metrics"].append(m)
    for cat, blob in by_category.items():
        scored = [m.score for m in blob["metrics"] if m.score is not None]
        blob["score"] = round(sum(scored) / len(scored), 2) if scored else None

    all_scored = [m.score for m in metrics if m.score is not None]
    overall = round(sum(all_scored) / len(all_scored), 2) if all_scored else 0.0

    return {
        "metrics": metrics,
        "by_category": by_category,
        "overall": overall,
        "scored": len(all_scored),
        "total": len(metrics),
    }


def format_scorecard(evaluation: dict) -> str:
    """Render the scorecard as readable text."""
    lines = [f"EVALUATION SCORECARD — overall {evaluation['overall']:.2f} "
             f"({evaluation['scored']}/{evaluation['total']} metrics scored)\n"]
    for cat, blob in evaluation["by_category"].items():
        cat_score = f"{blob['score']:.2f}" if blob["score"] is not None else "n/a"
        lines.append(f"\n{cat}  [{cat_score}]")
        for m in blob["metrics"]:
            s = f"{m.score:.2f}" if m.score is not None else " — "
            lines.append(f"  {m.code} {m.name}: {s}  ({m.note})")
    return "\n".join(lines)


# ============================================================================
# Helpers
# ============================================================================

def _extract_numbers(text: str) -> list:
    """Pull meaningful numbers (large values, percentages) from report text."""
    nums = []
    # Percentages like 39.3%
    for m in re.findall(r"(\d+(?:\.\d+)?)\s*%", text):
        nums.append(float(m))
    # Large numbers with commas like 281,724,000,000
    for m in re.findall(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", text):
        nums.append(float(m.replace(",", "")))
    # Decimals that look like ratios (e.g. P/E 25.4) — 2+ digit or with decimal
    for m in re.findall(r"\b(\d+\.\d+)\b", text):
        v = float(m)
        if v not in nums:
            nums.append(v)
    return nums


def _harvest_numbers(results: list) -> list:
    """Collect the numbers the agent actually gathered, for grounding checks."""
    out = []
    for r in results:
        obs = r.get("observation", {})
        if not obs.get("ok"):
            continue
        ratios = obs.get("ratios", {})
        for k, v in ratios.items():
            if isinstance(v, (int, float)):
                out.append(v)
                if abs(v) < 1:           # also store percentage form of ratios
                    out.append(round(v * 100, 2))
        income = obs.get("income_statement", {})
        for period in income.values():
            for v in period.values():
                if isinstance(v, (int, float)):
                    out.append(v)
        for r2 in obs.get("comparison", []):
            for v in r2.values():
                if isinstance(v, (int, float)):
                    out.append(v)
                    if abs(v) < 1:
                        out.append(round(v * 100, 2))
    return out


def _is_derivable(n: float, gathered: list) -> bool:
    """Loose check: is n close to any gathered number (handles rounding/scaling)?"""
    for g in gathered:
        if g == 0:
            continue
        if abs(n - g) < 0.5:
            return True
        ratio = n / g
        if 0.99 < ratio < 1.01:  # within 1%
            return True
    return False


_DIMENSION = {
    "sec_filing_search": "filing",
    "financial_data_api": "financials",
    "company_profile": "profile",
    "peer_comparison": "comparison",
    "web_search": "news",
    "news_sentiment": "news",
    "fact_checker": "verification",
    "earnings_transcript": "earnings",
    "calculation_engine": "computation",
    "vector_db_search": "memory",
}


def _dimension_of(tool: str) -> Optional[str]:
    return _DIMENSION.get(tool)


if __name__ == "__main__":
    # Demo on a canned report + results (no LLM needed for the deterministic ones).
    sample_report = (
        "# Investment Research Report — MSFT\n\n"
        "## Executive Summary\nMicrosoft is strong, with 39.3% margins.\n\n"
        "## Financial Highlights\nRevenue was 281,724,000,000 in FY2025; "
        "profit margin 39.3%, ROE 34.0%, trailing P/E 25.4.\n\n"
        "## Risks & Considerations\nHigh valuation.\n\n"
        "## Conclusion\nWell-positioned. This is not financial advice.\n\n"
        "## Sources\n- SEC EDGAR 10-K\n- yfinance"
    )
    sample_results = [
        {"tool": "sec_filing_search", "observation": {"ok": True, "url": "x"}},
        {"tool": "financial_data_api", "observation": {"ok": True, "ticker": "MSFT",
         "income_statement": {"2025": {"Total Revenue": 281724000000.0}},
         "ratios": {"profit_margin": 0.393, "return_on_equity": 0.34, "trailing_pe": 25.4}}},
    ]
    ev = evaluate_report(
        sample_report, results=sample_results,
        plan=[1, 2], sources=["SEC EDGAR 10-K", "yfinance"],
        use_llm_judge=False,
    )
    print(format_scorecard(ev))