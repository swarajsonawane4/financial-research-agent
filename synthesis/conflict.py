"""Conflict-resolution engine — reconcile disagreeing sources before synthesis.

When a research run gathers data from several sources (SEC filing, financial
API, news), they sometimes disagree about the same fact. This module decides
what to actually report, using the source-reliability hierarchy the registry
already defines (SOURCE_TIERS). It slots into the pipeline between EXECUTE and
SYNTHESIZE:

    plan -> execute -> [resolve conflicts] -> synthesize -> report

The resolution philosophy (see the design discussion):
  * Lower-tier sources (news, blogs) are largely DOWNSTREAM of primary sources —
    they report on the filing/earnings release. So when they disagree with a
    filing, the filing usually wins; the news is stale, mistaken, or spun.
  * NUMERIC conflicts -> trust the highest-tier source. Agreeing high-tier
    numbers are decisive.
  * QUALITATIVE conflicts (e.g. positive filing tone vs. negative news
    sentiment) -> do NOT force a winner. Both can be true (strong fundamentals,
    weak market sentiment). Surface the tension with context.
  * Never silently discard data: every resolved conflict is recorded so the
    report can say "the filing reports X; a news source cited Y; we use X."

This is deliberately LLM-assisted rather than a brittle numeric parser: the LLM
extracts comparable claims and judges whether they truly conflict, while the
TIER LOGIC (not the LLM) decides who wins, keeping resolution principled.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from dotenv import load_dotenv

from tools.tool_registry import tier_of

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None


# Map each TOOL to the source-type whose tier applies to its data.
# (SOURCE_TIERS is keyed by source-type; this connects tools to those types.)
TOOL_SOURCE_TYPE = {
    "sec_filing_search": "sec_filing",          # tier 1
    "financial_data_api": "financial_data_api",  # tier 2
    "company_profile": "financial_data_api",     # tier 2 (same provider family)
    "peer_comparison": "financial_data_api",     # tier 2
    "earnings_transcript": "earnings_call",      # tier 4
    "web_search": "major_news",                  # tier 3
    "news_sentiment": "major_news",              # tier 3
    "fact_checker": "major_news",                # tier 3 (web-evidence based)
}


def _llm_ready() -> bool:
    return genai is not None and bool(os.getenv("GEMINI_API_KEY"))


def _tool_tier(tool: str) -> int:
    """Reliability tier for a tool's data (lower = more reliable)."""
    return tier_of(TOOL_SOURCE_TYPE.get(tool, "social_forum"))


def detect_conflicts(results: list) -> dict:
    """Find disagreements among gathered findings and resolve them by tier.

    Args:
        results: the executor's results — a list of
                 {step, tool, args, observation} dicts.

    Returns a dict:
        {
          "ok": True,
          "conflicts": [
              {"topic": "...", "type": "numeric|qualitative",
               "claims": [{"tool":..., "tier":..., "value":...}, ...],
               "resolution": "...", "resolved_value": "..."|None,
               "unresolved": bool}
          ],
          "summary": "<one-line readout for the report>"
        }
    If the LLM is unavailable, returns an empty conflict set (the agent then
    synthesizes without explicit reconciliation — graceful degradation).
    """
    # Build a tier-tagged digest of successful findings for the LLM to inspect.
    findings = []
    for r in results:
        obs = r.get("observation", {})
        if not obs.get("ok"):
            continue
        tool = r.get("tool", "")
        findings.append({
            "tool": tool,
            "tier": _tool_tier(tool),
            "data": _compact(obs),
        })

    if len(findings) < 2 or not _llm_ready():
        # Need at least two sources to conflict; without the LLM we skip detection.
        return {"ok": True, "conflicts": [], "summary": "No conflicts detected."}

    decision = _llm_detect(findings)
    if not decision:
        return {"ok": True, "conflicts": [], "summary": "No conflicts detected."}

    # Apply tier-based resolution to each detected conflict (the LLM detects;
    # the TIER LOGIC here decides the winner, keeping it principled).
    resolved = []
    for c in decision.get("conflicts", []):
        resolved.append(_resolve_one(c))

    summary = (f"{len(resolved)} conflict(s) detected and reconciled."
               if resolved else "No conflicts detected.")
    return {"ok": True, "conflicts": resolved, "summary": summary}


def _resolve_one(conflict: dict) -> dict:
    """Apply the resolution rule to a single detected conflict."""
    claims = conflict.get("claims", [])
    ctype = conflict.get("type", "qualitative")
    topic = conflict.get("topic", "unknown")

    # Attach a tier to each claim (default to least-reliable if unknown tool).
    for c in claims:
        c["tier"] = _tool_tier(c.get("tool", ""))

    if not claims:
        return {"topic": topic, "type": ctype, "claims": claims,
                "resolution": "No claims to resolve.", "resolved_value": None,
                "unresolved": True}

    # Sort by tier (1 = most reliable first).
    claims_sorted = sorted(claims, key=lambda c: c["tier"])
    best = claims_sorted[0]
    best_tier = best["tier"]
    tied_at_best = [c for c in claims_sorted if c["tier"] == best_tier]

    if ctype == "numeric":
        if len(tied_at_best) == 1:
            # Clear winner: the single highest-tier numeric source.
            return {
                "topic": topic, "type": "numeric", "claims": claims_sorted,
                "resolution": (f"Numeric conflict: trusting the highest-reliability "
                               f"source ({best['tool']}, tier {best_tier}) over "
                               f"lower-tier sources."),
                "resolved_value": best.get("value"),
                "unresolved": False,
            }
        # Two+ equally-authoritative numeric sources disagree — genuinely unresolved.
        return {
            "topic": topic, "type": "numeric", "claims": claims_sorted,
            "resolution": (f"Numeric conflict between equally-authoritative sources "
                           f"(tier {best_tier}); reporting both — cannot adjudicate."),
            "resolved_value": None,
            "unresolved": True,
        }

    # Qualitative conflict: do NOT force a winner — surface the tension.
    return {
        "topic": topic, "type": "qualitative", "claims": claims_sorted,
        "resolution": ("Qualitative conflict (e.g. fundamentals vs. sentiment). "
                       "Both perspectives reported with context; not adjudicated, "
                       "as both may be valid simultaneously."),
        "resolved_value": None,
        "unresolved": True,
    }


def _compact(obs: dict) -> str:
    """A short string form of an observation for the LLM (keeps prompt small)."""
    # Reuse the same kind of compaction the synthesizer uses, abbreviated.
    if "filing_type" in obs:
        return f"{obs.get('company','')} {obs.get('filing_type','')} ({obs.get('filing_date','')})"
    if "ratios" in obs or "income_statement" in obs:
        bits = [f"ticker {obs.get('ticker','')}"]
        ratios = obs.get("ratios", {})
        for k in ("profit_margin", "revenue_growth", "return_on_equity"):
            if ratios.get(k) is not None:
                bits.append(f"{k}={ratios[k]*100:.1f}%")
        income = obs.get("income_statement", {})
        for p, vals in list(income.items())[:2]:
            if vals.get("Total Revenue"):
                bits.append(f"revenue {p}={vals['Total Revenue']:,.0f}")
        return "; ".join(bits)
    if obs.get("sentiment"):
        return f"news sentiment={obs['sentiment']} (score {obs.get('score')}): {obs.get('rationale','')}"
    if "results" in obs and obs.get("answer"):
        return f"news summary: {obs['answer'][:200]}"
    if "results" in obs:
        return "news headlines: " + "; ".join(r.get("title","") for r in obs["results"][:4])
    return str(obs)[:200]


def _llm_detect(findings: list) -> Optional[dict]:
    """Ask the LLM to extract conflicting claims across the findings."""
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    sources_block = "\n".join(
        f"- SOURCE {i+1} [{f['tool']}, reliability tier {f['tier']}]: {f['data']}"
        for i, f in enumerate(findings)
    )
    prompt = (
        "You are a financial analyst checking gathered research for CONFLICTS — "
        "places where two sources disagree about the same fact. Identify only "
        "REAL disagreements about the SAME underlying thing (same metric, or the "
        "same company's health). Ignore facts that simply appear in one source.\n\n"
        "For each conflict, classify it as:\n"
        "- 'numeric' — sources give different NUMBERS for the same metric, or\n"
        "- 'qualitative' — sources differ in TONE/assessment (e.g. one positive, "
        "one negative).\n\n"
        "Respond with ONLY JSON:\n"
        '{"conflicts": [{"topic": "<what they disagree on>", '
        '"type": "numeric|qualitative", "claims": [{"tool": "<source tool>", '
        '"value": "<that source\'s claim>"}]}]}\n'
        "If there are no real conflicts, return {\"conflicts\": []}.\n\n"
        f"GATHERED SOURCES:\n{sources_block}\n\nJSON:"
    )
    try:
        model = genai.GenerativeModel(DEFAULT_MODEL)
        text = (model.generate_content(prompt).text or "").strip()
        if text.startswith("```"):
            text = text.split("```")[1] if "```" in text[3:] else text
            text = text.replace("json", "", 1).strip().strip("`").strip()
        return json.loads(text)
    except Exception:  # noqa: BLE001
        return None


def conflicts_for_prompt(conflicts: list) -> str:
    """Render resolved conflicts as text to inject into the synthesis/report prompt."""
    if not conflicts:
        return "No source conflicts were detected."
    lines = ["Source conflicts detected and how they were reconciled:"]
    for c in conflicts:
        claim_str = "; ".join(
            f"{cl.get('tool','?')} (tier {cl.get('tier','?')}): {cl.get('value','?')}"
            for cl in c.get("claims", [])
        )
        lines.append(f"- On {c['topic']} [{c['type']}]: {claim_str}. "
                     f"Resolution: {c['resolution']}"
                     + (f" Trusted value: {c['resolved_value']}." if c.get("resolved_value") else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    # Demo with a canned qualitative conflict (filing positive vs news negative).
    demo = [
        {"tool": "financial_data_api", "observation": {
            "ok": True, "ticker": "PLTR",
            "ratios": {"profit_margin": 0.437, "revenue_growth": 0.847}}},
        {"tool": "news_sentiment", "observation": {
            "ok": True, "sentiment": "negative", "score": -0.4,
            "rationale": "Concerns about valuation and insider selling."}},
    ]
    print("Tier of financial_data_api:", _tool_tier("financial_data_api"))
    print("Tier of news_sentiment:", _tool_tier("news_sentiment"))
    out = detect_conflicts(demo)
    print(json.dumps(out, indent=2))