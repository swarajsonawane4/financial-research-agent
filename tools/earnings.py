"""Earnings tool — recent earnings commentary and highlights.

Honest scope note: full verbatim earnings-call transcripts require a paid data
provider (e.g. the major financial-data vendors). Free, reliable transcript
sources don't really exist. Rather than fake a transcript or ship a broken tool,
this tool gathers freely-available earnings *coverage* — recent reporting on the
company's latest results and management commentary — via the existing web search,
and clearly labels the output as a summary, not a verbatim transcript.

This still gives the agent useful signal for the "what did management say / how
did the quarter go" angle of a report, while being upfront about the limitation.

Entry point is `earnings_transcript`, matching the registry schema.
"""

from __future__ import annotations

from tools.web_search import web_search


def earnings_transcript(ticker: str, quarter: str = "", year: int = 0) -> dict:
    """Gather recent earnings coverage for a company.

    Args:
        ticker: stock ticker, e.g. "MSFT".
        quarter: optional fiscal quarter ("Q1".."Q4") to focus the search.
        year: optional fiscal year to focus the search.

    Returns a dict with ok flag, the search summary, and headlines. The result
    is explicitly labelled as earnings *coverage/summary*, not a verbatim
    transcript (see module docstring).
    """
    ticker = ticker.strip().upper()
    period = " ".join(p for p in [quarter, str(year) if year else ""] if p).strip()
    query = f"{ticker} earnings call {period} results highlights guidance".strip()

    search = web_search(query, num_results=8, date_range="past_year")
    if not search.get("ok"):
        return {
            "ok": False,
            "error": f"earnings_transcript needs web search: {search.get('error')}",
            "note": "Full verbatim transcripts require a paid provider; this tool "
                    "uses free earnings coverage instead.",
        }

    items = search.get("results", [])
    return {
        "ok": True,
        "ticker": ticker,
        "period": period or "latest",
        "data_type": "earnings_coverage_summary",  # NOT a verbatim transcript
        "note": "Summary of freely-available earnings coverage. For a verbatim "
                "transcript, a paid data provider would be required.",
        "summary": search.get("answer", ""),
        "headlines": [i.get("title", "") for i in items if i.get("title")][:8],
        "sources": [i.get("url") for i in items if i.get("url")][:5],
    }


if __name__ == "__main__":
    print("earnings_transcript requires a TAVILY_API_KEY. With it set, try:")
    print('  earnings_transcript("MSFT", "Q4", 2025)')
