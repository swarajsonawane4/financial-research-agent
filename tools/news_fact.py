"""News sentiment and fact-checking tools.

Two tools that build on the existing web_search (Tavily) plus the LLM — no new
dependencies:

  * news_sentiment — searches recent news for a company/topic and assesses the
                     overall tone (positive / negative / mixed / neutral) with a
                     short rationale. Useful for the "market sentiment" angle of
                     a research report.
  * fact_checker   — takes a specific claim, gathers evidence via web search, and
                     has the LLM judge whether the evidence supports it. This is
                     the agent's self-verification tool, used to reduce
                     hallucinated or stale claims before they reach the report.

Both degrade gracefully: if web search or the LLM is unavailable, they return a
clear, structured result rather than crashing.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from dotenv import load_dotenv

from tools.web_search import web_search

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None


def _llm_ready() -> bool:
    return genai is not None and bool(os.getenv("GEMINI_API_KEY"))


def _ask_llm_json(prompt: str) -> Optional[dict]:
    """Call the LLM and parse a JSON object from its reply, or None."""
    if not _llm_ready():
        return None
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(DEFAULT_MODEL)
        text = (model.generate_content(prompt).text or "").strip()
        if text.startswith("```"):
            text = text.split("```")[1] if "```" in text[3:] else text
            text = text.replace("json", "", 1).strip().strip("`").strip()
        return json.loads(text)
    except Exception:  # noqa: BLE001
        return None


def news_sentiment(query: str, num_articles: int = 8, lookback_days: int = 30) -> dict:
    """Assess the overall sentiment of recent news about a company/topic.

    Args:
        query: company name or topic.
        num_articles: how many articles to sample.
        lookback_days: how far back to look (passed to web search as a hint).

    Returns a dict with ok flag, an overall sentiment label, a -1..1 score, a
    short rationale, and the headlines considered.
    """
    date_range = "past_month" if lookback_days <= 31 else "past_year"
    search = web_search(f"{query} news", num_results=num_articles, date_range=date_range)
    if not search.get("ok"):
        return {"ok": False, "error": f"news_sentiment needs web search: {search.get('error')}",
                "results": []}

    articles = search.get("results", [])
    if not articles:
        return {"ok": True, "query": query, "sentiment": "neutral", "score": 0.0,
                "rationale": "No recent news found.", "headlines": []}

    headlines = [a.get("title", "") for a in articles if a.get("title")]
    snippets = "\n".join(f"- {a.get('title','')}: {a.get('snippet','')[:160]}" for a in articles)

    # Prefer LLM judgement; fall back to a simple keyword heuristic.
    decision = _ask_llm_json(
        "You are a financial news analyst. Assess the OVERALL sentiment of these "
        "headlines/snippets toward the subject. Respond with ONLY JSON: "
        '{"sentiment": "positive|negative|mixed|neutral", "score": <float -1..1>, '
        '"rationale": "<one sentence>"}\n\n'
        f"SUBJECT: {query}\n\nNEWS:\n{snippets}\n\nJSON:"
    )
    if decision and "sentiment" in decision:
        return {
            "ok": True, "query": query,
            "sentiment": decision.get("sentiment"),
            "score": decision.get("score"),
            "rationale": decision.get("rationale", ""),
            "headlines": headlines[:num_articles],
        }

    # Fallback heuristic: tiny positive/negative word count.
    pos = ("surge", "beat", "growth", "record", "gain", "up", "strong", "profit", "rise")
    neg = ("fall", "miss", "loss", "decline", "down", "weak", "lawsuit", "cut", "drop", "risk")
    text = snippets.lower()
    p = sum(text.count(w) for w in pos)
    n = sum(text.count(w) for w in neg)
    score = 0.0 if (p + n) == 0 else round((p - n) / (p + n), 2)
    label = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"
    return {"ok": True, "query": query, "sentiment": label, "score": score,
            "rationale": f"Keyword heuristic ({p} positive vs {n} negative signals).",
            "headlines": headlines[:num_articles]}


def fact_checker(claim: str, sources: Optional[list] = None) -> dict:
    """Verify a specific claim by gathering evidence and judging support.

    Args:
        claim: the specific statement to verify.
        sources: optional list of source strings already in hand; if omitted the
                 tool searches the web for evidence.

    Returns a dict with ok flag, a verdict (supported / refuted / unclear), a
    confidence, a short explanation, and the evidence considered.
    """
    # Gather evidence: use supplied sources, or search for them.
    if sources:
        evidence = "\n".join(f"- {s}" for s in sources)
        evidence_urls = []
    else:
        search = web_search(claim, num_results=6)
        if not search.get("ok"):
            return {"ok": False, "error": f"fact_checker needs web search: {search.get('error')}"}
        items = search.get("results", [])
        evidence = "\n".join(f"- {i.get('title','')}: {i.get('snippet','')[:200]}" for i in items)
        evidence_urls = [i.get("url") for i in items if i.get("url")]
        if search.get("answer"):
            evidence = f"- Summary: {search['answer']}\n" + evidence

    if not evidence.strip():
        return {"ok": True, "claim": claim, "verdict": "unclear", "confidence": 0.0,
                "explanation": "No evidence found to assess the claim.", "evidence_urls": []}

    decision = _ask_llm_json(
        "You are a fact-checker. Given a CLAIM and EVIDENCE, judge whether the "
        "evidence supports the claim. Respond with ONLY JSON: "
        '{"verdict": "supported|refuted|unclear", "confidence": <float 0..1>, '
        '"explanation": "<one or two sentences citing the evidence>"}\n\n'
        f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence}\n\nJSON:"
    )
    if decision and "verdict" in decision:
        return {
            "ok": True, "claim": claim,
            "verdict": decision.get("verdict"),
            "confidence": decision.get("confidence"),
            "explanation": decision.get("explanation", ""),
            "evidence_urls": evidence_urls[:5],
        }

    # Without the LLM we can't judge nuance; return the evidence for a human.
    return {"ok": True, "claim": claim, "verdict": "unclear", "confidence": 0.0,
            "explanation": "LLM unavailable; evidence gathered but not judged.",
            "evidence_urls": evidence_urls[:5]}


if __name__ == "__main__":
    print("news_sentiment and fact_checker require a TAVILY_API_KEY (and GEMINI_API_KEY")
    print("for best results). With keys set, try:")
    print('  news_sentiment("NVIDIA")')
    print('  fact_checker("Microsoft FY2025 revenue exceeded $280 billion")')
