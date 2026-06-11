# Challenge Results

The agent was validated across 8 progressive research challenges of increasing
difficulty, each targeting a specific capability. All 8 completed successfully.
Run them with `python -m challenges.run_challenges`.

| # | Challenge | Capability proven | Outcome |
|---|-----------|-------------------|---------|
| 1 | Company Profile (Microsoft) | Multi-tool planning + synthesis | ✓ Profiled business segments, strategy, and financials into one report |
| 2 | Financial Deep-Dive (Apple) | Structured financial retrieval + interpretation | ✓ Analyzed revenue trend, margins, ROE, valuation |
| 3 | Peer Comparison (NVIDIA) | LLM-supplied competitor tickers | ✓ Compared NVDA vs AMD, INTC, AVGO across valuation + profitability |
| 4 | Risk Assessment (Tesla) | Risk extraction across sources | ✓ Pulled risks from filing + recent news |
| 5 | The Contradiction (Palantir) | Conflict resolution: fundamentals vs. sentiment | ✓ Held the tension — strong financials alongside mixed market sentiment, explained rather than forced |
| 6 | Ambiguous Query ("the banks") | Planning under vague input | ✓ Decomposed into identify-banks → sector news → major players → sentiment |
| 7 | Memory / Cross-Session | Long-term memory recall | ✓ Researched Microsoft, then recalled the stored findings on a follow-up without re-fetching |
| 8 | Stress Test (NVIDIA, 50% tool failure) | Resilience under failure | ✓ With financial_data_api, web_search, and news_sentiment forcibly failing, the agent retried, fell back (financial_data_api → company_profile), tripped the circuit breaker on dead tools, and still produced a report |

## Highlights

**Challenge 8 (stress test)** is the strongest demonstration of robustness. Half
the data tools were deliberately disabled mid-run. The resilience trace shows the
agent retrying the failed tool three times, falling back to an alternative, and
opening the circuit breaker on tools confirmed dead — then completing a report
from the tools that remained (company profile, SEC filing, peer comparison). The
agent degraded gracefully instead of crashing.

**Challenge 5 (the contradiction)** exercises the conflict-resolution engine. The
agent gathered Palantir's financials, news, sentiment, and profile, then
synthesized a balanced view: strong fundamentals (43.7% profit margin, 84.7%
revenue growth) alongside mixed market sentiment (valuation debate, contract
scrutiny) — presenting the tension with context rather than forcing a single
verdict, which is how a careful analyst reasons.

**Challenge 7 (memory)** proves the agent learns across runs: it researched
Microsoft's financials in one run (storing the findings to the vector database),
then on a follow-up query recognized it already knew the answer and recalled it
from memory rather than re-fetching from the API.

## Notes

- Source reliability follows a corrected tier hierarchy (SEC filings > financial
  APIs > major news > earnings calls > social/forums) used for conflict
  resolution. See `tools/tool_registry.py` and `ERROR_LOG.md`.
- Verbatim earnings-call transcripts require a paid data provider; the
  `earnings_transcript` tool gathers freely-available earnings coverage instead
  and labels its output accordingly.