"""Minimal agent loop — Day 1 skeleton.

This is a deliberately small Thought -> Action -> Observation loop built on
LangGraph. Its only job today is to prove the plumbing works end to end: the
agent receives a query, decides to call the SEC EDGAR tool, executes it, and
reports back with REAL data from the SEC.

It runs in two modes:
  * LLM mode  — if an LLM is available (Gemini/OpenAI/Anthropic/Ollama), it can
                later drive tool selection. Wired loosely here; expanded on Day 4.
  * Mock mode — if no LLM key is configured, a tiny rule-based "planner" decides
                the action. This lets you SEE the loop work today without paying
                for or setting up an LLM. Mock mode is clearly labelled in output.

Run it:
    python -m agent.core "Get Microsoft's latest 10-K"
or just:
    python -m agent.core
"""

from __future__ import annotations

import os
import re
import sys
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

# The real tool registry + schemas (Day 2).
from tools.tool_registry import Tool, ToolRegistry
from tools.schemas import (
    SEC_FILING_SEARCH, FINANCIAL_DATA_API, WEB_SEARCH,
    VECTOR_DB_SEARCH, VECTOR_DB_STORE,
    COMPANY_PROFILE, PEER_COMPARISON, NEWS_SENTIMENT,
    FACT_CHECKER, CALCULATION_ENGINE, EARNINGS_TRANSCRIPT,
)
from tools.sec_edgar import sec_filing_search
from tools.financial_api import financial_data_api
from tools.web_search import web_search
from tools.company import company_profile, peer_comparison
from tools.news_fact import news_sentiment, fact_checker
from tools.analysis import calculation_engine
from tools.earnings import earnings_transcript
from memory.vector_store import vector_db_search, vector_db_store
from agent.llm_client import (
    llm_available, llm_decide_tool, llm_make_plan, llm_synthesize, DEFAULT_MODEL,
)
from synthesis.conflict import detect_conflicts, conflicts_for_prompt
from agent.error_handler import call_with_resilience, CircuitBreaker


# --- Shared state passed between graph nodes ---------------------------------

class AgentState(TypedDict):
    query: str            # the user's research query
    thought: str          # the agent's reasoning about what to do
    action: dict          # the chosen tool call: {"tool": str, "args": dict}
    observation: dict     # the tool's returned result
    answer: str           # the final natural-language answer
    plan: list            # multi-step plan: list of step dicts
    results: list         # collected results, one per executed step
    conflicts: list       # conflicts detected/resolved across sources
    report: dict          # paths to the generated report files


# --- Tool registry -----------------------------------------------------------
# Build the registry and register tools. More tools get added here on Day 5/7;
# new tools can also be unlocked at runtime (the progression-unlock mechanic).

REGISTRY = ToolRegistry()
REGISTRY.register(
    Tool(
        name="sec_filing_search",
        description=(
            "Retrieve official SEC EDGAR filings (10-K, 10-Q, 8-K, DEF 14A) for "
            "a US public company by ticker. Use for authoritative regulatory data."
        ),
        parameters=SEC_FILING_SEARCH,
        fn=sec_filing_search,
        tier=1,  # SEC filings are the most reliable source
    )
)
REGISTRY.register(
    Tool(
        name="financial_data_api",
        description=(
            "Retrieve structured financials (income statement, balance sheet, "
            "cash flow, key ratios) for a company by ticker. Use for numbers: "
            "revenue, margins, P/E, growth rates."
        ),
        parameters=FINANCIAL_DATA_API,
        fn=financial_data_api,
        tier=2,  # curated structured data
        fallbacks=["company_profile"],  # if financials fail, at least get basic profile
    )
)
REGISTRY.register(
    Tool(
        name="web_search",
        description=(
            "Search the web for current news and commentary about a company or "
            "topic. Use for recent developments that post-date the latest filing."
        ),
        parameters=WEB_SEARCH,
        fn=web_search,
        tier=3,  # professional news
    )
)
REGISTRY.register(
    Tool(
        name="vector_db_search",
        description=(
            "Search the agent's long-term memory for findings from PAST research. "
            "Check this FIRST before external calls — the agent may already know."
        ),
        parameters=VECTOR_DB_SEARCH,
        fn=vector_db_search,
    )
)
REGISTRY.register(
    Tool(
        name="vector_db_store",
        description=(
            "Store a research finding in long-term memory for future retrieval. "
            "Use after producing a verified, citable finding."
        ),
        parameters=VECTOR_DB_STORE,
        fn=vector_db_store,
    )
)
REGISTRY.register(
    Tool(
        name="company_profile",
        description=(
            "Get a company's qualitative profile by ticker: sector, industry, "
            "business summary, employees, country, website. Use to understand "
            "what a company does."
        ),
        parameters=COMPANY_PROFILE,
        fn=company_profile,
        tier=2,
    )
)
REGISTRY.register(
    Tool(
        name="peer_comparison",
        description=(
            "Compare a company against peer tickers on key valuation and "
            "profitability metrics (P/E, margins, growth). You MUST pass "
            "peers=['TICKER1','TICKER2',...] — supply the competitor tickers "
            "yourself from your knowledge (e.g. for PLTR: SNOW, DDOG, MDB). The "
            "tool does not find peers automatically."
        ),
        parameters=PEER_COMPARISON,
        fn=peer_comparison,
        tier=2,
    )
)
REGISTRY.register(
    Tool(
        name="news_sentiment",
        description=(
            "Assess the overall sentiment (positive/negative/mixed) of recent "
            "news about a company or topic, with a score and rationale."
        ),
        parameters=NEWS_SENTIMENT,
        fn=news_sentiment,
        tier=3,
        fallbacks=["web_search"],  # if sentiment scoring fails, fall back to raw news
    )
)
REGISTRY.register(
    Tool(
        name="fact_checker",
        description=(
            "Verify a specific factual claim by gathering evidence and judging "
            "whether it is supported, refuted, or unclear. Use to self-check facts."
        ),
        parameters=FACT_CHECKER,
        fn=fact_checker,
        tier=2,
    )
)
REGISTRY.register(
    Tool(
        name="calculation_engine",
        description=(
            "Perform exact financial calculations: growth_rate, cagr, margin, "
            "pe_ratio, roe, ev_ebitda, dcf. Pass inputs as a dict. Use for any "
            "arithmetic rather than computing it yourself."
        ),
        parameters=CALCULATION_ENGINE,
        fn=calculation_engine,
    )
)
REGISTRY.register(
    Tool(
        name="earnings_transcript",
        description=(
            "Gather recent earnings coverage and management-commentary highlights "
            "for a company (summary of free sources, not a verbatim transcript)."
        ),
        parameters=EARNINGS_TRANSCRIPT,
        fn=earnings_transcript,
        tier=4,
        fallbacks=["web_search"],  # earnings coverage falls back to general news
    )
)


# --- Node 1: THINK (decide what to do) ---------------------------------------

def _detect_ticker(q: str) -> str:
    """Pull a ticker out of an uppercased query (name match, then heuristic)."""
    name_to_ticker = {
        "MICROSOFT": "MSFT", "APPLE": "AAPL", "NVIDIA": "NVDA", "TESLA": "TSLA",
        "AMAZON": "AMZN", "GOOGLE": "GOOGL", "ALPHABET": "GOOGL", "META": "META",
        "NETFLIX": "NFLX", "PALANTIR": "PLTR",
    }
    for name, sym in name_to_ticker.items():
        if name in q:
            return sym
    stopwords = {
        "GET", "THE", "SHOW", "ME", "A", "AN", "FOR", "OF", "IS", "WHAT",
        "PULL", "FIND", "LATEST", "RECENT", "ANNUAL", "REPORT", "FILING",
        "THEIR", "UP", "TO", "AND", "ON", "IN", "ANALYZE", "ANALYSE",
        "NEWS", "REVENUE", "FINANCIALS", "ABOUT", "WITH", "HAPPENING",
    }
    candidates = [
        tok for tok in re.findall(r"\b([A-Z]{1,5})\b", q) if tok not in stopwords
    ]
    return candidates[0] if candidates else "MSFT"


def _mock_planner(query: str) -> tuple[str, dict]:
    """Rule-based stand-in for an LLM planner.

    Routes the query to one of three registered tools based on intent keywords.
    This is intentionally simple; on Day 4 an LLM replaces it and will chain
    multiple tools together rather than picking just one.
    """
    q = query.upper()
    ticker = _detect_ticker(q)

    # --- Route by intent ---
    memory_words = ("WHAT DO I KNOW", "WHAT HAVE I", "ALREADY RESEARCHED",
                    "REMEMBER", "RECALL", "FROM MEMORY", "PREVIOUSLY")
    financial_words = ("REVENUE", "FINANCIAL", "INCOME", "BALANCE", "CASH FLOW",
                       "MARGIN", "PROFIT", "EARNINGS PER", "RATIO", "P/E", "PE ")
    news_words = ("NEWS", "HAPPENING", "LATEST ON", "RECENT", "WHAT'S NEW",
                  "DEVELOPMENT", "SENTIMENT")

    if any(w in q for w in memory_words):
        thought = (
            "The user is asking what I already know. I'll search long-term "
            "memory (vector_db_search) before making any external calls."
        )
        # If a ticker was detected, filter to it; otherwise search broadly.
        filt = {"ticker": ticker} if ticker != "MSFT" or "MICROSOFT" in q else None
        action = {"tool": "vector_db_search", "args": {"query": query, "top_k": 5}}
        if filt:
            action["args"]["filter"] = filt
        return thought, action

    if any(w in q for w in financial_words):
        thought = (
            f"The user wants financial data for {ticker}. "
            f"I'll call financial_data_api to retrieve the statements and ratios."
        )
        action = {
            "tool": "financial_data_api",
            "args": {"ticker": ticker, "statement_type": "all", "period": "annual", "years": 3},
        }
        return thought, action

    if any(w in q for w in news_words):
        thought = (
            f"The user wants recent news about {ticker}. "
            f"I'll call web_search for current coverage."
        )
        action = {"tool": "web_search", "args": {"query": f"{ticker} stock news", "num_results": 8}}
        return thought, action

    # default: SEC filing
    filing_type = "10-K"
    for ft in ("10-K", "10-Q", "8-K", "DEF 14A"):
        if ft in q:
            filing_type = ft
            break
    thought = (
        f"The user wants a {filing_type} filing for {ticker}. "
        f"I'll call sec_filing_search to retrieve it from EDGAR."
    )
    action = {"tool": "sec_filing_search", "args": {"ticker": ticker, "filing_type": filing_type}}
    return thought, action


def think(state: AgentState) -> AgentState:
    """Decide which tool to call.

    LLM-first: ask the language model to choose the tool and arguments. If the
    LLM is unavailable (no key, network/quota error, unparseable response), fall
    back to the keyword planner so the agent still works. The fallback is the
    same graceful-degradation pattern used throughout the system.
    """
    query = state["query"]

    # 1. Try the LLM planner.
    decision = None
    if llm_available():
        decision = llm_decide_tool(query, REGISTRY.describe())

    # 2. Fall back to the keyword planner if the LLM didn't give a usable answer.
    if decision and decision.get("tool") and REGISTRY.has(decision["tool"]):
        thought = decision.get("thought", f"Calling {decision['tool']}.")
        action = {"tool": decision["tool"], "args": decision.get("args", {})}
        mode = f"LLM planner ({DEFAULT_MODEL})"
    else:
        thought, action = _mock_planner(query)
        mode = "keyword fallback (LLM unavailable)" if llm_available() else "MOCK MODE (no LLM key)"

    print(f"\n[THINK]  ({mode})")
    print(f"  Thought: {thought}")
    print(f"  Action:  {action['tool']}({action['args']})")

    state["thought"] = thought
    state["action"] = action
    return state


# --- Node 2: ACT (execute the tool) ------------------------------------------

def act(state: AgentState) -> AgentState:
    """Run the chosen tool via the registry (which validates args first)."""
    action = state["action"]

    print("\n[ACT]")
    # The registry validates the call against the tool's schema, then dispatches.
    observation = REGISTRY.call(action["tool"], action["args"])

    print(f"  Observation: {observation}")
    state["observation"] = observation
    return state


# --- Node 3: ANSWER (summarize for the user) ---------------------------------

def answer(state: AgentState) -> AgentState:
    """Turn the observation into a short human-readable answer.

    Formats differently depending on which tool produced the result. (On Day 7
    this becomes the LLM-driven report generator; for now it's a clean summary.)
    """
    obs = state["observation"]
    tool = state["action"]["tool"]

    if not obs.get("ok"):
        ans = f"I couldn't complete that: {obs.get('error', 'unknown error')}"

    elif tool == "sec_filing_search":
        ans = (
            f"{obs['company']} ({obs['ticker']}) — most recent {obs['filing_type']} "
            f"was filed on {obs['filing_date']}.\n"
            f"  Accession: {obs['accession']}\n"
            f"  Document:  {obs['url']}"
        )

    elif tool == "financial_data_api":
        lines = [f"Financial data for {obs['ticker']} ({obs['period']}):"]
        income = obs.get("income_statement", {})
        if income:
            periods = list(income.keys())
            lines.append(f"  Income statement periods: {', '.join(periods)}")
            for p in periods:
                rev = income[p].get("Total Revenue")
                if rev:
                    lines.append(f"    {p}: Total Revenue = {rev:,.0f}")

        ratios = obs.get("ratios", {})
        if ratios:
            lines.append("  Key ratios:")
            # Format each ratio nicely. Percentages get *100 and a % sign;
            # plain multiples (P/E, P/B) get one decimal; market cap gets commas.
            def _fmt_pct(v):
                return f"{v * 100:.1f}%" if v is not None else None

            display = [
                ("Profit margin", _fmt_pct(ratios.get("profit_margin"))),
                ("Return on equity", _fmt_pct(ratios.get("return_on_equity"))),
                ("Revenue growth", _fmt_pct(ratios.get("revenue_growth"))),
                ("Trailing P/E", f"{ratios['trailing_pe']:.1f}" if ratios.get("trailing_pe") else None),
                ("Forward P/E", f"{ratios['forward_pe']:.1f}" if ratios.get("forward_pe") else None),
                ("Price/Book", f"{ratios['price_to_book']:.1f}" if ratios.get("price_to_book") else None),
                ("Debt/Equity", f"{ratios['debt_to_equity']:.1f}" if ratios.get("debt_to_equity") else None),
                ("Market cap", f"{ratios['market_cap']:,.0f}" if ratios.get("market_cap") else None),
            ]
            for label, value in display:
                if value is not None:
                    lines.append(f"    {label}: {value}")
        ans = "\n".join(lines)

        # Store a concise finding in long-term memory so the agent can recall it
        # later. Works whether the query returned income data, ratios, or both.
        try:
            from memory.vector_store import vector_db_store
            margin = ratios.get("profit_margin") if ratios else None
            latest = list(income.keys())[0] if income else None
            rev = income[latest].get("Total Revenue") if latest else None

            finding_bits = []
            if rev:
                finding_bits.append(f"most recent reported revenue (period {latest}) was {rev:,.0f}")
            if margin is not None:
                finding_bits.append(f"profit margin of {margin * 100:.1f}%")
            pe = ratios.get("trailing_pe") if ratios else None
            if pe:
                finding_bits.append(f"trailing P/E of {pe:.1f}")

            if finding_bits:
                finding = f"{obs['ticker']} " + ", ".join(finding_bits) + "."
                vector_db_store(finding, {
                    "ticker": obs["ticker"],
                    "source_type": "financial_data_api",
                    "confidence": 0.95,
                })
                lines.append("  (stored this finding in long-term memory)")
                ans = "\n".join(lines)
        except Exception:  # noqa: BLE001 - storing is best-effort, never block the answer
            pass

    elif tool == "vector_db_search":
        results = obs.get("results", [])
        if not results:
            ans = ("Long-term memory has nothing relevant yet. "
                   "Research a company first, then ask again.")
        else:
            lines = [f"From long-term memory ({len(results)} finding(s)):"]
            for r in results:
                lines.append(
                    f"  [similarity {r['similarity']}] ({r['ticker']}, "
                    f"{r['source_type']}, {r['date']}): {r['content']}"
                )
            ans = "\n".join(lines)

    elif tool == "web_search":
        results = obs.get("results", [])
        lines = [f"Web search: '{obs.get('query', '')}' — {len(results)} results"]
        if obs.get("answer"):
            lines.append(f"  Summary: {obs['answer']}")
        for r in results[:5]:
            lines.append(f"  - {r['title']}\n    {r['url']}")
        ans = "\n".join(lines)

    else:
        ans = str(obs)

    print("\n[ANSWER]")
    print("  " + ans.replace("\n", "\n  "))
    state["answer"] = ans
    return state


# --- Build the graph ---------------------------------------------------------

# --- Multi-step Plan-and-Execute nodes ---------------------------------------
# These three nodes implement the fuller agent: PLAN a list of steps, EXECUTE
# each one collecting results, then SYNTHESIZE everything into one answer. They
# reuse the registry and the single-tool helpers. If the LLM can't produce a
# plan, plan_node falls back to a single-step plan via the keyword planner, so
# the agent always has something to execute.

def plan_node(state: AgentState) -> AgentState:
    """PLAN: ask the LLM for an ordered list of tool-call steps."""
    query = state["query"]

    plan = None
    if llm_available():
        plan = llm_make_plan(query, REGISTRY.describe())

    # Fallback: wrap the single-tool decision as a one-step plan.
    if not plan:
        thought, action = _mock_planner(query)
        plan = [{"step": 1, "thought": thought, "tool": action["tool"], "args": action["args"]}]
        source = "keyword fallback (single step)"
    else:
        source = f"LLM planner ({DEFAULT_MODEL})"

    # Keep only steps whose tool actually exists in the registry.
    plan = [s for s in plan if REGISTRY.has(s["tool"])]
    if not plan:  # everything was hallucinated; final safety net
        thought, action = _mock_planner(query)
        plan = [{"step": 1, "thought": thought, "tool": action["tool"], "args": action["args"]}]

    print(f"\n[PLAN]  ({source}) — {len(plan)} step(s)")
    for s in plan:
        print(f"  {s['step']}. {s['tool']}({s['args']})  — {s['thought']}")

    state["plan"] = plan
    state["results"] = []
    return state


def execute_node(state: AgentState) -> AgentState:
    """EXECUTE: run each planned step with resilience (retry, fallback, breaker)."""
    print("\n[EXECUTE]")
    results = []
    breaker = CircuitBreaker(failure_threshold=2)  # shared across this run's steps
    for s in state["plan"]:
        observation = call_with_resilience(REGISTRY, s["tool"], s["args"], breaker=breaker)
        ok = observation.get("ok")
        # Surface when resilience kicked in (retry/fallback/breaker), for transparency.
        trace = observation.get("_resilience")
        status = "ok" if ok else "FAILED"
        if trace and any("falling back" in t for t in trace):
            status += " (via fallback)"
        print(f"  step {s['step']} {s['tool']}: {status}")
        if ok:
            REGISTRY.mark_useful()
        results.append({
            "step": s["step"],
            "tool": s["tool"],
            "args": s["args"],
            "observation": observation,
        })
        # Auto-store useful financial findings in long-term memory.
        _maybe_store_finding(s["tool"], observation)

    # Report any tools the circuit breaker took down this run.
    down = breaker.status()
    if down:
        print(f"  (circuit breaker — tools down this run: {down})")

    state["results"] = results
    return state


def resolve_node(state: AgentState) -> AgentState:
    """RESOLVE: detect and reconcile conflicts across sources before synthesis.

    Uses the source-reliability hierarchy: numeric conflicts resolve to the
    highest-tier source; qualitative conflicts (fundamentals vs. sentiment) are
    surfaced rather than forced. The result feeds into synthesis and the report.
    """
    print("\n[RESOLVE]")
    outcome = detect_conflicts(state["results"])
    conflicts = outcome.get("conflicts", [])
    state["conflicts"] = conflicts

    if not conflicts:
        print("  No source conflicts detected.")
    else:
        for c in conflicts:
            tag = "unresolved" if c.get("unresolved") else f"-> {c.get('resolved_value')}"
            print(f"  conflict on '{c['topic']}' [{c['type']}] {tag}")
    return state


def synthesize_node(state: AgentState) -> AgentState:
    """SYNTHESIZE: combine all collected results into one coherent answer.

    Now conflict-aware: the reconciliation summary is passed to the LLM so the
    synthesis reflects which sources were trusted and surfaces genuine tensions.
    """
    query = state["query"]
    results = state["results"]
    conflicts = state.get("conflicts", [])

    # Build a compact text digest of everything gathered, for the LLM to synthesize.
    digest_parts = []
    for r in results:
        obs = r["observation"]
        digest_parts.append(
            f"[Step {r['step']} — {r['tool']}({r['args']})]\n{_summarize_observation(obs)}"
        )
    gathered = "\n\n".join(digest_parts)

    # Append the conflict reconciliation so synthesis respects it.
    conflict_text = conflicts_for_prompt(conflicts)
    gathered_with_conflicts = f"{gathered}\n\n--- SOURCE RECONCILIATION ---\n{conflict_text}"

    # Try LLM synthesis; fall back to showing the structured digest.
    synthesized = llm_synthesize(query, gathered_with_conflicts) if llm_available() else None
    if synthesized:
        ans = synthesized
        mode = f"LLM synthesis ({DEFAULT_MODEL})"
    else:
        ans = "Gathered findings:\n\n" + gathered_with_conflicts
        mode = "raw digest (LLM unavailable)"

    print(f"\n[SYNTHESIZE]  ({mode})")
    print("  " + ans.replace("\n", "\n  "))
    state["answer"] = ans
    return state


def report_node(state: AgentState) -> AgentState:
    """REPORT: write a structured investment research report (Markdown + PDF)."""
    query = state["query"]
    results = state["results"]

    # Rebuild the gathered digest (same shape synthesis used).
    digest_parts = []
    for r in results:
        obs = r["observation"]
        digest_parts.append(
            f"[Step {r['step']} — {r['tool']}({r['args']})]\n{_summarize_observation(obs)}"
        )
    gathered = "\n\n".join(digest_parts)

    # Include conflict reconciliation in the report's source material.
    conflicts = state.get("conflicts", [])
    if conflicts:
        gathered += "\n\n--- SOURCE RECONCILIATION ---\n" + conflicts_for_prompt(conflicts)
    ticker = ""
    sources = []
    for r in results:
        args = r.get("args", {})
        if not ticker and isinstance(args.get("ticker"), str):
            ticker = args["ticker"]
        obs = r["observation"]
        if not obs.get("ok"):
            continue
        if r["tool"] == "sec_filing_search" and obs.get("url"):
            sources.append(f"SEC EDGAR {obs.get('filing_type','')} — {obs['url']}")
        elif r["tool"] == "financial_data_api":
            sources.append(f"Financial data (yfinance) — {obs.get('ticker','')}")
        elif r["tool"] == "web_search":
            for item in obs.get("results", [])[:3]:
                if item.get("url"):
                    sources.append(item["url"])

    try:
        from synthesis.report import generate_report
        result = generate_report(query, gathered, ticker=ticker, sources=sources, make_pdf=True)
        print("\n[REPORT]")
        print(f"  Markdown: {result.get('markdown_path')}")
        if result.get("pdf_path"):
            print(f"  PDF:      {result.get('pdf_path')}")
        else:
            print("  PDF:      (not generated — see note above)")
        state["report"] = result
    except Exception as exc:  # noqa: BLE001 - report is the finale; don't crash the run
        print(f"\n[REPORT]  skipped: {type(exc).__name__}: {exc}")
        state["report"] = {"ok": False, "error": str(exc)}
    return state


# --- Helpers shared by the multi-step nodes ----------------------------------

def _summarize_observation(obs: dict) -> str:
    """Compact, readable summary of a single tool observation for synthesis."""
    if not obs.get("ok"):
        return f"FAILED: {obs.get('error', 'unknown error')}"

    # SEC filing
    if "filing_type" in obs and "filing_date" in obs:
        return (f"{obs.get('company', '')} {obs['filing_type']} filed "
                f"{obs['filing_date']} — {obs.get('url', '')}")

    # Financial data
    if "ratios" in obs or "income_statement" in obs:
        bits = [f"Ticker {obs.get('ticker', '')}"]
        income = obs.get("income_statement", {})
        for p, vals in list(income.items())[:3]:
            rev = vals.get("Total Revenue")
            if rev:
                bits.append(f"Revenue {p}: {rev:,.0f}")
        ratios = obs.get("ratios", {})
        if ratios:
            for k in ("profit_margin", "return_on_equity", "revenue_growth"):
                v = ratios.get(k)
                if v is not None:
                    bits.append(f"{k}: {v * 100:.1f}%")
            for k in ("trailing_pe", "forward_pe", "debt_to_equity"):
                v = ratios.get(k)
                if v is not None:
                    bits.append(f"{k}: {v:.1f}")
            if ratios.get("market_cap"):
                bits.append(f"market_cap: {ratios['market_cap']:,.0f}")
        return "; ".join(bits)

    # Web search
    if "results" in obs and isinstance(obs["results"], list):
        if obs.get("answer"):
            return f"News summary: {obs['answer']}"
        heads = [f"- {r.get('title', '')}" for r in obs["results"][:5]]
        return "News headlines:\n" + "\n".join(heads) if heads else "No news results."

    # Memory recall
    if "results" in obs:
        return f"Memory returned {len(obs['results'])} finding(s)."

    return str(obs)


def _maybe_store_finding(tool: str, obs: dict) -> None:
    """Store a concise finding in long-term memory after a useful data fetch."""
    if tool != "financial_data_api" or not obs.get("ok"):
        return
    try:
        ratios = obs.get("ratios", {})
        income = obs.get("income_statement", {})
        latest = list(income.keys())[0] if income else None
        rev = income[latest].get("Total Revenue") if latest else None
        margin = ratios.get("profit_margin")
        bits = []
        if rev:
            bits.append(f"most recent reported revenue (period {latest}) was {rev:,.0f}")
        if margin is not None:
            bits.append(f"profit margin of {margin * 100:.1f}%")
        if bits:
            finding = f"{obs['ticker']} " + ", ".join(bits) + "."
            vector_db_store(finding, {
                "ticker": obs["ticker"],
                "source_type": "financial_data_api",
                "confidence": 0.95,
            })
    except Exception:  # noqa: BLE001
        pass


def build_agent():
    """Wire the Plan-and-Execute graph: plan -> execute -> synthesize."""
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("resolve", resolve_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "resolve")
    graph.add_edge("resolve", "synthesize")
    graph.add_edge("synthesize", "report")
    graph.add_edge("report", END)

    return graph.compile()


# --- Entry point -------------------------------------------------------------

def main(query: Optional[str] = None) -> None:
    query = query or "Give me a profile of Microsoft"
    print("=" * 64)
    print(f"QUERY: {query}")
    print("=" * 64)

    agent = build_agent()
    agent.invoke({"query": query})

    print("\n" + "=" * 64)
    print("Done.  (plan -> execute -> resolve -> synthesize -> report)")
    print("=" * 64)


if __name__ == "__main__":
    user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    main(user_query)