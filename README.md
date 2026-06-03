# Autonomous Financial Research Agent

An autonomous AI agent that researches a public company end-to-end — planning its
own multi-step research, pulling data from SEC EDGAR filings, financial APIs, and
live web search, and synthesizing the findings into a structured investment
research report (Markdown + PDF), without step-by-step human guidance.

Built on a **Plan-and-Execute** reasoning loop with LangGraph: the agent reads a
query, writes a multi-step plan, executes each step through a validated tool
registry, recovers gracefully when a source fails, and synthesizes everything
into a professional report.

```
Query: "Give me a profile of Microsoft"

[PLAN]       3 steps - 10-K filing, financial statements, recent news
[EXECUTE]    step 1 ok - step 2 ok - step 3 ok
[SYNTHESIZE] one coherent analyst writeup
[REPORT]     results/report_MSFT_2026-06-03.{md,pdf}
```

## What works today

- **Plan-and-Execute loop** — an LLM (Gemini) decomposes a query into an ordered
  multi-step plan, then executes the steps and synthesizes the results.
- **Tool registry with schema validation** — tools are called through a central
  registry that validates arguments against each tool's schema *before*
  dispatch, so a malformed or hallucinated tool call fails cleanly instead of
  crashing. Five tools are live: SEC EDGAR filings, financial data (yfinance),
  web search (Tavily), and long-term memory read/write.
- **Three-layer memory** — short-term context, long-term semantic memory
  (Chroma vector DB, persists across runs), and an episodic log of what worked.
  Research a company once and the agent recalls the findings on later runs.
- **Structured report generation** — produces an investment research report with
  named sections (executive summary, financials, developments, risks,
  conclusion) and cited sources, as both Markdown and PDF.
- **Graceful degradation** — if a data source is unavailable, the agent uses
  what it has and reports the gap honestly rather than fabricating or crashing.

## Architecture

```
Query
  -> PLAN        LLM writes an ordered multi-step research plan
  -> EXECUTE     each step runs through the validated tool registry:
                   SEC EDGAR / financial data / web search / memory R-W
  -> SYNTHESIZE  results are combined into one coherent analysis
  -> REPORT      structured investment report (Markdown + PDF)

Memory (spanning all runs):
  short-term (context) / long-term (Chroma vectors) / episodic (JSON log)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Set the SEC User-Agent to your own name + email in tools/sec_edgar.py
#    (the SEC requires this; it rejects requests without it).

# 2. Add a free Gemini API key for the planner:
cp .env.example .env          # then put your key in the GEMINI_API_KEY line
#    (get one free at https://aistudio.google.com/apikey)
#    Optional: add a free TAVILY_API_KEY for web search.

# 3. Run a research query:
python -m agent.core "Give me a profile of Microsoft"
```

The agent will print its plan, execute each step against real data sources,
synthesize an answer, and write a report to `results/`. (Without a Gemini key it
still runs, falling back to a keyword-based planner.)

## Tech stack

- **Orchestration:** LangGraph (Plan-and-Execute state machine)
- **LLM:** Google Gemini (free tier); falls back to a keyword planner if no key
- **Vector DB:** Chroma (local) with sentence-transformers embeddings
- **Data sources:** SEC EDGAR (no key), yfinance, Tavily web search
- **Reporting:** reportlab (PDF, pure-Python)

## Roadmap

The core research pipeline is working. Planned next:

- More tools (earnings transcripts, news sentiment, peer comparison,
  fact-checker, calculator) to expand the registry past 10
- A multi-source synthesis engine with explicit conflict resolution, built on
  the source-reliability hierarchy already in the registry
- Formal error handling with retry/backoff and fallback chains
- An evaluation framework (20+ quality metrics)
- Validation across a set of progressive research challenges

## Notes

This project generates analysis for informational purposes only; it is not
financial advice. It was built independently as a portfolio project.
