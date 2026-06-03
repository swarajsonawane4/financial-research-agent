"""Episodic memory — a process-level log of the agent's past experiences.

Where the vector store remembers *facts* (what the agent learned), episodic
memory remembers *experiences* (what the agent did and how it went): which tools
it used for a given query type, which strategies worked, what errors it hit, and
how it recovered. Over time this lets the agent improve its planning — e.g. "for
risk-assessment queries, earnings transcripts have been the most useful source,
so prioritize them."

This is deliberately simple: an append-only JSON-lines file on disk. Each line
is one episode. No database needed; it's a log, and logs are easy to reason
about and inspect by hand.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG_DIR = Path.home() / ".cache" / "financial-research-agent"
LOG_FILE = LOG_DIR / "episodic_memory.jsonl"


class EpisodicMemory:
    """Append-only log of research episodes."""

    def __init__(self, log_file: Optional[Path] = None) -> None:
        self.log_file = log_file or LOG_FILE
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        query: str,
        query_type: str = "",
        tools_used: Optional[list[str]] = None,
        tools_useful: Optional[list[str]] = None,
        errors: Optional[list[str]] = None,
        outcome: str = "",
        notes: str = "",
    ) -> dict:
        """Append one episode to the log.

        Args:
            query: the research query handled.
            query_type: classification, e.g. "company_profile", "risk_assessment".
            tools_used: every tool the agent called.
            tools_useful: the subset whose results actually made it into the report.
            errors: error strings encountered (and presumably recovered from).
            outcome: short result summary, e.g. "completed", "partial", "failed".
            notes: free-text lesson, e.g. "web_search was redundant; filing had it".
        """
        episode = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "query_type": query_type,
            "tools_used": tools_used or [],
            "tools_useful": tools_useful or [],
            "errors": errors or [],
            "outcome": outcome,
            "notes": notes,
        }
        with self.log_file.open("a") as f:
            f.write(json.dumps(episode) + "\n")
        return {"ok": True, "recorded": episode["timestamp"]}

    def all_episodes(self) -> list[dict]:
        """Return every recorded episode."""
        if not self.log_file.exists():
            return []
        episodes = []
        for line in self.log_file.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return episodes

    def recall_strategy(self, query_type: str) -> dict:
        """Suggest a strategy for a query type, based on past episodes.

        Looks at prior episodes of the same type and reports which tools were
        most often *useful*, so the planner can prioritize them. This is the
        agent 'learning from experience' in a simple, inspectable way.
        """
        relevant = [e for e in self.all_episodes() if e.get("query_type") == query_type]
        if not relevant:
            return {"ok": True, "query_type": query_type, "episodes": 0, "suggested_tools": []}

        # tally how often each tool was useful for this query type
        tally: dict[str, int] = {}
        for e in relevant:
            for tool in e.get("tools_useful", []):
                tally[tool] = tally.get(tool, 0) + 1

        ranked = sorted(tally.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "ok": True,
            "query_type": query_type,
            "episodes": len(relevant),
            "suggested_tools": [tool for tool, _ in ranked],
            "tool_usefulness": dict(ranked),
        }

    def common_errors(self) -> dict:
        """Tally the most common errors across all episodes."""
        tally: dict[str, int] = {}
        for e in self.all_episodes():
            for err in e.get("errors", []):
                key = err.split(":")[0]  # group by error type prefix
                tally[key] = tally.get(key, 0) + 1
        ranked = sorted(tally.items(), key=lambda kv: kv[1], reverse=True)
        return {"ok": True, "errors": dict(ranked)}


if __name__ == "__main__":
    # Smoke test: record a couple of episodes, then recall a strategy.
    epi = EpisodicMemory()
    epi.record(
        query="Risk assessment for Tesla",
        query_type="risk_assessment",
        tools_used=["sec_filing_search", "web_search", "earnings_transcript"],
        tools_useful=["sec_filing_search", "earnings_transcript"],
        outcome="completed",
        notes="earnings transcript surfaced forward-looking risks the filing missed",
    )
    epi.record(
        query="Risk assessment for Ford",
        query_type="risk_assessment",
        tools_used=["sec_filing_search", "earnings_transcript", "news_sentiment"],
        tools_useful=["sec_filing_search", "earnings_transcript", "news_sentiment"],
        outcome="completed",
    )
    print("Total episodes:", len(epi.all_episodes()))
    print("Strategy for 'risk_assessment':")
    print(" ", epi.recall_strategy("risk_assessment"))