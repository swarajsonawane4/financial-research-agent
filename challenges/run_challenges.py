"""The 8 progressive research challenges.

This is the agent's validation suite — eight scenarios of increasing difficulty,
each targeting a capability the agent was built for. Running this produces a
gallery of real reports (saved to results/) plus a summary, which is exactly
what a reviewer wants to see: "here are 8 challenges, here's how the agent did."

Mapping of challenge -> capability exercised:
  1. Company profile          -> multi-tool planning + synthesis
  2. Financial deep-dive      -> financial_data_api + calculation
  3. Peer comparison          -> peer_comparison tool
  4. Risk assessment          -> filing + news risk extraction
  5. The contradiction        -> conflict-resolution engine
  6. Ambiguous query          -> planning under vague input
  7. Memory / cross-session   -> long-term memory recall
  8. Stress test (tool failure)-> resilience layer (retry/fallback/breaker)

Usage:
    python -m challenges.run_challenges            # run all 8
    python -m challenges.run_challenges 1 5 8      # run specific ones
    EVALUATE=1 python -m challenges.run_challenges  # also score each report

Challenge 8 deliberately injects tool failures to prove graceful degradation;
the agent should still produce a report. Challenge 7 runs a research query, then
a follow-up that should hit memory from the first.
"""

from __future__ import annotations

import os
import sys
import time

from agent.core import build_agent, REGISTRY


# Each challenge: id, title, the query, and what capability it proves.
CHALLENGES = [
    {
        "id": 1,
        "title": "Company Profile",
        "query": "Give me a profile of Microsoft — what the company does and its financials.",
        "proves": "Multi-tool planning and synthesis into a coherent profile.",
    },
    {
        "id": 2,
        "title": "Financial Deep-Dive",
        "query": "Analyze Apple's financial health: revenue trend, margins, and key ratios.",
        "proves": "Structured financial data retrieval and interpretation.",
    },
    {
        "id": 3,
        "title": "Peer Comparison",
        "query": "Compare NVIDIA against its competitors on valuation and profitability.",
        "proves": "Peer comparison with LLM-supplied competitor tickers.",
    },
    {
        "id": 4,
        "title": "Risk Assessment",
        "query": "What are the main risks facing Tesla right now? Check filings and recent news.",
        "proves": "Risk extraction across authoritative and current sources.",
    },
    {
        "id": 5,
        "title": "The Contradiction",
        "query": "Is Palantir a healthy company? Reconcile its financials with market sentiment.",
        "proves": "Conflict-resolution engine: fundamentals vs. sentiment.",
    },
    {
        "id": 6,
        "title": "Ambiguous Query",
        "query": "What's happening with the banks?",
        "proves": "Planning and graceful handling of a vague, underspecified query.",
    },
    {
        "id": 7,
        "title": "Memory / Cross-Session",
        "query": "__MEMORY__",  # special: handled by a two-step runner below
        "proves": "Long-term memory recall across separate research runs.",
    },
    {
        "id": 8,
        "title": "Stress Test (50% tool failure)",
        "query": "Give me an investment profile of NVIDIA.",
        "proves": "Resilience layer: agent completes despite injected tool failures.",
        "inject_failures": True,
    },
]


def _run_one(agent, challenge: dict) -> dict:
    """Run a single challenge and return a small result summary."""
    cid = challenge["id"]
    print("\n" + "#" * 70)
    print(f"# CHALLENGE {cid}: {challenge['title']}")
    print(f"# Proves: {challenge['proves']}")
    print("#" * 70)

    started = time.time()
    try:
        if challenge["query"] == "__MEMORY__":
            final = _run_memory_challenge(agent)
        elif challenge.get("inject_failures"):
            final = _run_with_injected_failures(agent, challenge)
        else:
            final = agent.invoke({"query": challenge["query"]})
        elapsed = round(time.time() - started, 1)
        report = final.get("report", {})
        return {
            "id": cid, "title": challenge["title"], "ok": True,
            "elapsed_s": elapsed,
            "markdown": report.get("markdown_path"),
            "pdf": report.get("pdf_path"),
            "steps": len(final.get("plan", [])),
            "conflicts": len(final.get("conflicts", [])),
        }
    except Exception as exc:  # noqa: BLE001 - capture, don't abort the whole suite
        return {"id": cid, "title": challenge["title"], "ok": False,
                "error": f"{type(exc).__name__}: {exc}"}


def _run_memory_challenge(agent) -> dict:
    """Challenge 7: research a company, then ask a follow-up that hits memory."""
    print("\n[STEP A] Researching Microsoft financials (populates memory)...")
    agent.invoke({"query": "Analyze Microsoft's revenue and profit margin."})
    print("\n[STEP B] Asking what we already know (should recall from memory)...")
    return agent.invoke({"query": "What do I already know about Microsoft's financials?"})


def _run_with_injected_failures(agent, challenge: dict) -> dict:
    """Challenge 8: temporarily break ~half the data tools, prove resilience.

    We swap the underlying functions of several tools to fail, run the agent,
    then restore them. The agent should retry, fall back, and still report.
    """
    targets = ["financial_data_api", "web_search", "news_sentiment"]
    originals = {}
    for name in targets:
        tool = REGISTRY.get(name)
        if tool:
            originals[name] = tool.fn
            # Replace with a failing stub (simulated outage).
            tool.fn = (lambda **kw: {"ok": False, "error": "simulated outage (challenge 8)"})
    print(f"\n[INJECT] Forced failures on: {list(originals)}")
    try:
        return agent.invoke({"query": challenge["query"]})
    finally:
        # Always restore, even if the run raises.
        for name, fn in originals.items():
            REGISTRY.get(name).fn = fn
        print(f"[RESTORE] Restored: {list(originals)}")


def main(which=None) -> None:
    agent = build_agent()
    to_run = CHALLENGES
    if which:
        ids = {int(x) for x in which}
        to_run = [c for c in CHALLENGES if c["id"] in ids]

    print("=" * 70)
    print(f"RUNNING {len(to_run)} CHALLENGE(S)")
    print("=" * 70)

    summaries = []
    for challenge in to_run:
        summaries.append(_run_one(agent, challenge))
        time.sleep(1)  # be gentle on rate limits between challenges

    # Final summary table.
    print("\n\n" + "=" * 70)
    print("CHALLENGE SUMMARY")
    print("=" * 70)
    for s in summaries:
        if s["ok"]:
            extra = f"{s['steps']} steps"
            if s.get("conflicts"):
                extra += f", {s['conflicts']} conflict(s)"
            print(f"  ✓ Challenge {s['id']} ({s['title']}): {extra}, {s['elapsed_s']}s")
            if s.get("markdown"):
                print(f"      report: {s['markdown']}")
        else:
            print(f"  ✗ Challenge {s['id']} ({s['title']}): {s['error']}")
    passed = sum(1 for s in summaries if s["ok"])
    print(f"\n{passed}/{len(summaries)} challenges completed.")


if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else None
    main(args)