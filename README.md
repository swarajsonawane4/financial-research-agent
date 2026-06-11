# Autonomous Financial Research Agent

An autonomous AI agent that researches a public company end-to-end — planning its
own multi-step research, gathering data from SEC EDGAR filings, financial APIs,
news, and earnings coverage, reconciling conflicting sources, and synthesizing
the findings into a structured investment research report (Markdown + PDF) —
without step-by-step human guidance.

Built on a **Plan-and-Execute** reasoning loop with LangGraph. The agent reads a
query, writes a multi-step plan, executes each step through a validated tool
registry, resolves conflicts between sources, recovers gracefully when tools
fail, and synthesizes a professional report.

```
Query: "Is Palantir a healthy company? Reconcile financials with market sentiment."

[PLAN]       4 steps — financials, news, sentiment, company profile
[EXECUTE]    step 1 ok · step 2 ok · step 3 ok · step 4 ok
[RESOLVE]    reconciled sources by reliability tier
[SYNTHESIZE] strong fundamentals (43.7% margin, 84.7% growth) vs. mixed sentiment
[REPORT]     results/report_PLTR_2026-06-11.{md,pdf}
```

## Capabilities

- **Plan-and-Execute loop** — an LLM (Gemini) decomposes a query into an ordered
  multi-step plan, executes each step, and synthesizes the results. Falls back to
  a keyword planner if the LLM is unavailable, so the agent always functions.
- **Tool registry (11 tools) with schema validation** — every tool call is
  validated against its JSON schema *before* dispatch, so malformed or
  hallucinated calls fail cleanly instead of crashing. Tools: SEC EDGAR filings,
  financial data (yfinance), web search (Tavily), company profile, peer
  comparison, news sentiment, fact-checker, calculation engine, earnings
  coverage, and long-term memory read/write.
- **Three-layer memory** — short-term context, long-term semantic memory (Chroma
  vector DB, persists across runs), and an episodic log. Research a company once
  and the agent recalls the findings on later runs.
- **Conflict-resolution engine** — when sources disagree, numeric conflicts
  resolve to the highest-reliability source (SEC filing > financial API > news),
  and qualitative tensions (strong fundamentals vs. weak sentiment) are surfaced
  with context rather than forced to a false verdict.
- **Resilience layer** — retry with exponential backoff, per-tool fallback chains
  (e.g. financial data → company profile), and a circuit breaker that skips a
  dead tool after repeated failures. The agent completes a useful report even
  with half its tools down.
- **Structured report generation** — investment reports with named sections
  (executive summary, financials, developments, risks, conclusion) and cited
  sources, as both Markdown and PDF.
- **Evaluation framework (21 metrics)** — automated scoring across five
  categories: accuracy, completeness, source quality, agent behavior, and report
  quality (LLM-as-judge). Mostly deterministic, so evaluation runs without API
  cost.
- **Validated across 8 progressive challenges** — see `challenges/RESULTS.md`.

## Architecture

```
Query
  → PLAN        LLM writes an ordered multi-step research plan
  → EXECUTE     each step runs through the validated tool registry, with
                  retry / fallback / circuit-breaker resilience
  → RESOLVE     conflicting sources reconciled by reliability tier
  → SYNTHESIZE  results combined into one coherent analysis
  → REPORT      structured investment report (Markdown + PDF)

Memory (spanning all runs):
  short-term (context) · long-term (Chroma vectors) · episodic (JSON log)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Set the SEC User-Agent to your own name + email in tools/sec_edgar.py
#    (the SEC requires this; it rejects requests without it).

# 2. Add a free Gemini API key for the planner:
cp .env.example .env          # then put your key in the GEMINI_API_KEY line
#    (free key at https://aistudio.google.com/apikey)
#    Optional: add a free TAVILY_API_KEY for web search.

# 3. Run a research query:
python -m agent.core "Give me a profile of Microsoft"

# Run with self-evaluation (prints a 21-metric scorecard):
EVALUATE=1 python -m agent.core "Give me a profile of Microsoft"

# Run the 8 progressive validation challenges:
python -m challenges.run_challenges          # all 8
python -m challenges.run_challenges 1 5 8    # specific ones
```

## Tests

Deterministic test suites (no API key or network required):

```bash
python -m tests.test_conflict          # conflict-resolution engine (5 tests)
python -m tests.test_error_handling    # resilience layer (6 tests)
python -m tests.test_evaluation        # evaluation framework (6 tests)
```

## Tech stack

- **Orchestration:** LangGraph (Plan-and-Execute state machine)
- **LLM:** Google Gemini (free tier); falls back to a keyword planner without a key
- **Vector DB:** Chroma (local) with sentence-transformers embeddings
- **Data sources:** SEC EDGAR (no key), yfinance, Tavily web search
- **Reporting:** reportlab (PDF, pure-Python)

## Project structure

```
agent/        core pipeline (plan→execute→resolve→synthesize→report),
              LLM client, resilience layer
tools/        the 11 tools + registry with schema validation
memory/       Chroma vector store + episodic log
synthesis/    report generator + conflict-resolution engine
evaluation/   21-metric evaluation framework
challenges/   the 8 progressive validation challenges
tests/        deterministic test suites
results/      generated reports (Markdown + PDF)
```

## Notes

This project generates analysis for informational purposes only; it is not
financial advice. It was built independently as a portfolio project.