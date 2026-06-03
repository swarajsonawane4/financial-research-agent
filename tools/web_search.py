"""Web search tool — recent news and commentary via Tavily."""

from __future__ import annotations

import os
from typing import Optional

import httpx

TAVILY_URL = "https://api.tavily.com/search"


def web_search(
    query: str,
    num_results: int = 10,
    date_range: Optional[str] = None,
) -> dict:
    """Search the web for current news / commentary."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "error": "TAVILY_API_KEY not set. Get a free key at tavily.com and "
            "add it to your .env. (The agent can proceed without web search; "
            "fallback chains will route around it.)",
            "results": [],
        }

    days = None
    if date_range:
        days = {"past_week": 7, "past_month": 30, "past_year": 365}.get(date_range)

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max(1, min(num_results, 20)),
        "search_depth": "advanced",
        "include_answer": True,
    }
    if days:
        payload["days"] = days

    try:
        resp = httpx.post(TAVILY_URL, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
                "score": r.get("score"),
            }
            for r in data.get("results", [])
        ]
        return {
            "ok": True,
            "query": query,
            "answer": data.get("answer", ""),
            "results": results,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "results": []}
