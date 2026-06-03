# ERROR_LOG.md

The reference project document states it contains exactly **7 deliberate
factual or logical errors** (page 17, "Error Detection Exercise"). This file
documents the errors identified, with explanations and the correct version.
Where an error affects implementation, the corrected behaviour is noted and
flagged in the relevant source file.

> Confidence note: errors 1–6 are high-confidence. The 7th is the hardest to pin
> down; the strongest candidate is documented, but each should be re-verified
> against current primary sources rather than taken on faith — that verification
> is itself part of the exercise.

---

## Error 1 — Memory Utilization metric (AB-4) formula contradiction
**Location:** Section A5.2, Category 5, metric AB-4 (page 19).
**The error:** AB-4 is defined as "the ratio of memory hits to total external
API calls," but the same bullet then says it is "calculated as memory_hits
**multiplied by** total_api_calls."
**Why it's wrong:** A ratio is a division, not a multiplication. Multiplying the
two values produces a number that grows without bound as call volume rises and
has no meaningful interpretation as a utilization rate.
**Correct version:** `memory_utilization = memory_hits / external_api_calls`.
**Where corrected in code:** `tools/tool_registry.py`,
`ToolRegistry.memory_utilization()`.

---

## Error 2 — Source-reliability hierarchy ranks forums above major news
**Location:** Section A6.2, "Tiered Source Trustworthiness" (page 21).
**The error:** The hierarchy lists "Tier 4: Social media posts and anonymous
forum discussions" as MORE reliable than "Tier 5: Major news outlets (Reuters,
Bloomberg News, Financial Times)."
**Why it's wrong:** Reputable, editorially-overseen journalism is more reliable
than anonymous, unverified, manipulation-prone social media. The ordering is
inverted; professional news should rank above social/forum sources.
**Correct version:** SEC filings > financial data APIs > major news >
earnings-call transcripts > social/forum. (Earnings calls are direct but subject
to management spin, so they sit below independent journalism for factual
reliability.)
**Where corrected in code:** `tools/tool_registry.py`, `SOURCE_TIERS`.

---

## Error 3 — US bank stress tests dated to 2007 / attributed to Dodd-Frank
**Location:** Section A7.3, "Query Disambiguation" (page 24).
**The error:** "The first US bank stress tests under SCAP were conducted in 2007
following the Dodd-Frank Act."
**Why it's wrong:** Two problems. (a) The Supervisory Capital Assessment Program
(SCAP) was conducted in **2009**, not 2007 — the preceding sentence in the same
paragraph even says 2009 correctly, contradicting this one. (b) The
Dodd-Frank Act was enacted in **2010**, so it could not have prompted a stress
test in 2007 or 2009. SCAP predated Dodd-Frank.
**Correct version:** SCAP was conducted in 2009; the recurring, Dodd-Frank-
mandated stress tests (CCAR / DFAST) came after Dodd-Frank's 2010 passage.

---

## Error 4 — Indian companies file "Form 20-F" with the MCA
**Location:** Case Study 4, Section C4.2 (page 42).
**The error:** "Indian companies file annual returns using Form 20-F with the
MCA (Ministry of Corporate Affairs), similar to the 10-K filing in the US."
**Why it's wrong:** Form 20-F is a **US SEC** form, filed by foreign private
issuers listing in the United States — it is not an Indian MCA filing. Indian
companies file annual returns/financials with the MCA on forms such as MGT-7
(annual return) and AOC-4 (financial statements).
**Correct version:** Indian annual MCA filings use forms like MGT-7 and AOC-4;
Form 20-F is the SEC filing for foreign issuers in the US.

---

## Error 5 — text-embedding-3-large dimensions / cost inconsistency
**Location:** Section E2.2, "Recommended Embedding Models" (page 62).
**The error:** "OpenAI text-embedding-3-large: 1024 dimensions ... 6.5x more
expensive."
**Why it's wrong:** text-embedding-3-large produces **3072** dimensions, not
1024. The figure is also internally inconsistent: it lists "large" at 1024 dims,
which is *fewer* than the "small" model's 1536 dims listed just above, yet calls
large the higher-quality, more expensive model — a larger/better embedding model
having fewer dimensions than the smaller one makes no sense.
**Correct version:** text-embedding-3-small = 1536 dims; text-embedding-3-large
= 3072 dims.

---

## Error 6 — fabricated "45–60%" hallucination-rate statistic
**Location:** Case Study 3, Section C3.2 (page 40).
**The error:** "Industry average hallucination rates for unverified financial
agents are typically around 45-60%."
**Why it's wrong:** This is presented as an established statistic but is
implausible and unsupported — it would imply roughly half of all claims from an
unverified agent are fabricated, which is far higher than observed behaviour and
is not a recognized industry benchmark. It reads as a planted fabricated figure
(notably more extreme than the 23% the same case study attributes to the agent
itself).
**Correct version:** No credible single "industry average" of 45–60% exists;
hallucination rates vary widely by model, task, and prompting, and unverified
rates on factual financial tasks are materially lower than half.

---

## Error 7 — (lower confidence — verify against primary sources)
**Strongest candidate:** Section A6.2 / A6.3 internal tension, or a second
embedding-spec detail. Several formula definitions in the glossary were checked
and are CORRECT and should NOT be logged as errors:
- EBITDA Margin = EBITDA / Revenue ✓
- Free Cash Flow = Operating Cash Flow − CapEx ✓
- ROE = Net Income / Shareholders' Equity ✓
- Basis points = 0.01% (100 bps = 1 percentage point) ✓
- Market Cap = Share Price × Shares Outstanding ✓
- BloombergGPT = 50B parameters ✓ (matches the published model)

**Action:** Re-read Parts A–E once more specifically hunting the 7th, and verify
each candidate against a primary source before committing it here. Do not pad
the list to reach 7 with a weak claim — a wrong "error" costs credibility.
