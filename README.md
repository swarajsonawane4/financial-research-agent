# Autonomous Financial Research Agent

An autonomous AI agent that researches a public company end-to-end — pulling SEC
filings, financial data, news, and earnings-call transcripts — and synthesizes
the findings into a structured investment research report, without step-by-step
human guidance.

Built with a **Plan-and-Execute** reasoning loop on LangGraph, a registry of 10+
tools, a three-layer memory system (short-term / vector / episodic), a
multi-source synthesis engine that resolves conflicting data, fallback-chain
error handling, and an evaluation framework spanning 20+ quality metrics —
validated across 8 progressive research challenges.

> Status: **in active development.** Day 1 ships a working Thought → Action →
> Observation loop that retrieves real SEC EDGAR filings. Day 2 adds a
> structured tool registry (12-tool schemas, schema-validated dispatch,
> fallback chains) and a corrected source-reliability hierarchy.

## Architecture (target)

```
Query
  -> Plan-and-Execute loop (planner -> executor -> observer -> replanner)
  -> Tool registry (10+ tools): SEC EDGAR, financial APIs, web search,
       earnings transcripts, news, calculator, memory R/W, ...
  -> Three-layer memory: short-term (context) / long-term (Chroma vectors) / episodic (JSON)
  -> Multi-source synthesis (source-reliability hierarchy + conflict resolution)
  -> Structured investment research report (Markdown / PDF)
  -> Evaluation framework (20+ metrics, 8 challenges)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Edit the User-Agent in tools/sec_edgar.py to your own name + email (SEC requires this).

# Run the Day 1 loop — works with NO LLM key (mock mode):
python -m agent.core "Get Microsoft's latest 10-K"
```

Expected output: the agent reasons about the query, calls the SEC EDGAR API, and
reports the real filing date, accession number, and document URL.

## Tech stack

- **Orchestration:** LangGraph
- **LLM:** Gemini / OpenAI / Anthropic / Ollama (configurable; runs without one in mock mode)
- **Vector DB:** Chroma (local) + sentence-transformers embeddings
- **Data:** SEC EDGAR (no key), yfinance, Tavily, Financial Modeling Prep

## Roadmap

See the 15-day build plan. Phases: foundation -> core brain (loop + memory +
tools) -> synthesis + error handling -> evaluation + challenges -> docs + demo.
