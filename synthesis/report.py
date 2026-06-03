"""Report generator — turns gathered research into a structured report.

This is the agent's final deliverable: not a chat answer, but a structured
investment research report with named sections (executive summary, company
overview, financial highlights, recent developments, risks, conclusion) and a
sources list. It produces:

  1. Markdown — the canonical, version-controllable form (renders on GitHub).
  2. PDF      — a polished render of the same content, via reportlab (pure
                Python, installs cleanly on macOS — no system libraries needed).

The LLM is asked to fill the section structure from the gathered data, citing
sources. If the LLM is unavailable, a plain structured report is still produced
from the raw findings, so the agent always yields a usable document.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None

# Where reports are written.
REPORTS_DIR = Path("results")

# The section structure of an investment research report.
SECTIONS = [
    "Executive Summary",
    "Company Overview",
    "Financial Highlights",
    "Recent Developments",
    "Risks & Considerations",
    "Conclusion",
]


def _llm_available() -> bool:
    return genai is not None and bool(os.getenv("GEMINI_API_KEY"))


def _llm_write_report(query: str, ticker: str, gathered: str) -> Optional[str]:
    """Ask the LLM to write the report body as structured Markdown sections."""
    if not _llm_available():
        return None

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    section_list = "\n".join(f"  - {s}" for s in SECTIONS)
    system_instruction = (
        "You are a junior financial analyst writing an investment research "
        "report. Using ONLY the gathered data provided, write the report body "
        "in Markdown with these exact sections, each as a level-2 heading "
        f"(##):\n{section_list}\n\n"
        "Rules:\n"
        "- Use only facts present in the gathered data; never invent numbers.\n"
        "- Executive Summary: 2-3 sentences capturing the key takeaway.\n"
        "- Financial Highlights: present the concrete figures (revenue, margins, "
        "ratios) clearly; a small Markdown table is good here.\n"
        "- Recent Developments: summarize news findings; if none, say so.\n"
        "- Risks & Considerations: note real risks evident from the data, and "
        "flag where data was missing or sources were unavailable.\n"
        "- Conclusion: a brief, balanced wrap-up. This is analysis, NOT financial "
        "advice — do not tell the reader to buy or sell.\n"
        "- Keep it professional and concise. Output ONLY the Markdown body, "
        "starting at the first ## heading."
    )

    prompt = (
        f"{system_instruction}\n\n"
        f"RESEARCH QUERY: {query}\n"
        f"PRIMARY TICKER: {ticker or 'N/A'}\n\n"
        f"GATHERED DATA:\n{gathered}\n\n"
        "Report body (Markdown):"
    )

    try:
        model = genai.GenerativeModel(DEFAULT_MODEL)
        resp = model.generate_content(prompt)
        return (resp.text or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


def _fallback_body(gathered: str) -> str:
    """A plain structured report when the LLM is unavailable."""
    return (
        "## Executive Summary\n\n"
        "Automated summary unavailable (LLM not configured). Raw findings below.\n\n"
        "## Financial Highlights\n\n"
        f"{gathered}\n\n"
        "## Conclusion\n\n"
        "This report was assembled from the gathered data without narrative "
        "synthesis. See the findings above."
    )


def generate_report(
    query: str,
    gathered: str,
    *,
    ticker: str = "",
    sources: Optional[list] = None,
    make_pdf: bool = True,
) -> dict:
    """Produce a structured research report as Markdown (and optionally PDF).

    Args:
        query: the original research query.
        gathered: the digest of gathered findings (from the execute step).
        ticker: primary ticker, used in the title and filename.
        sources: list of source strings to cite (URLs, tool names).
        make_pdf: also render a PDF alongside the Markdown.

    Returns:
        dict with ok flag and the paths written:
        {"ok": True, "markdown_path": "...", "pdf_path": "..."|None}
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    title_subject = ticker.upper() if ticker else "Research"
    title = f"Investment Research Report — {title_subject}"

    # Body: LLM-written sections, or a structured fallback.
    body = _llm_write_report(query, ticker, gathered) or _fallback_body(gathered)

    # Assemble the full Markdown document.
    header = (
        f"# {title}\n\n"
        f"*Prepared: {today}*  \n"
        f"*Query: {query}*\n\n"
        "---\n\n"
    )
    sources = sources or []
    sources_md = ""
    if sources:
        sources_md = "\n\n## Sources\n\n" + "\n".join(f"- {s}" for s in sources)

    disclaimer = (
        "\n\n---\n\n*This report was generated by an autonomous research agent "
        "for informational purposes only and does not constitute financial advice.*\n"
    )

    markdown = header + body + sources_md + disclaimer

    # Write Markdown.
    stem = f"report_{title_subject}_{today}".replace(" ", "_")
    md_path = REPORTS_DIR / f"{stem}.md"
    md_path.write_text(markdown)

    pdf_path = None
    if make_pdf:
        try:
            pdf_path = _render_pdf(markdown, REPORTS_DIR / f"{stem}.pdf", title)
        except Exception as exc:  # noqa: BLE001 - PDF is a bonus; never block the MD
            print(f"  (PDF render skipped: {type(exc).__name__}: {exc})")

    return {
        "ok": True,
        "markdown_path": str(md_path),
        "pdf_path": str(pdf_path) if pdf_path else None,
    }


def _render_pdf(markdown: str, out_path: Path, title: str) -> Path:
    """Render the Markdown report to a clean PDF using reportlab.

    This is a lightweight Markdown-to-PDF: it handles headings (#, ##),
    bullet lists, simple tables, and paragraphs — enough for our report
    structure. reportlab is pure Python and installs without system libraries.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportH1", parent=styles["Title"], fontSize=18, spaceAfter=6))
    styles.add(ParagraphStyle(name="ReportH2", parent=styles["Heading2"], fontSize=13,
                              textColor=colors.HexColor("#1a3c5e"), spaceBefore=12, spaceAfter=4))
    styles.add(ParagraphStyle(name="ReportBody", parent=styles["Normal"], fontSize=10,
                              leading=15, spaceAfter=6))
    styles.add(ParagraphStyle(name="ReportMeta", parent=styles["Normal"], fontSize=9,
                              textColor=colors.grey, spaceAfter=2))

    doc = SimpleDocTemplate(
        str(out_path), pagesize=letter,
        topMargin=0.8 * inch, bottomMargin=0.8 * inch,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        title=title,
    )

    def _md_inline(text: str) -> str:
        """Convert simple Markdown inline (**bold**) to reportlab markup."""
        import re
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
        return text

    story = []
    lines = markdown.split("\n")
    i = 0
    table_buffer: list = []

    def _flush_table():
        nonlocal table_buffer
        if not table_buffer:
            return
        # Parse Markdown table rows into a reportlab Table.
        rows = []
        for row in table_buffer:
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            # skip separator rows like |---|---|
            if all(set(c) <= set("-: ") for c in cells):
                continue
            rows.append([Paragraph(_md_inline(c), styles["ReportBody"]) for c in cells])
        if rows:
            t = Table(rows, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t)
            story.append(Spacer(1, 8))
        table_buffer = []

    while i < len(lines):
        line = lines[i].rstrip()

        if line.startswith("|") and line.endswith("|"):
            table_buffer.append(line)
            i += 1
            continue
        else:
            _flush_table()

        if not line:
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(_md_inline(line[2:]), styles["ReportH1"]))
        elif line.startswith("## "):
            story.append(Paragraph(_md_inline(line[3:]), styles["ReportH2"]))
        elif line.startswith("### "):
            story.append(Paragraph(_md_inline(line[4:]), styles["ReportH2"]))
        elif line.startswith("---"):
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
            story.append(Spacer(1, 4))
        elif line.startswith("*") and line.endswith("*") and not line.startswith("**"):
            story.append(Paragraph(_md_inline(line), styles["ReportMeta"]))
        elif line.startswith("- ") or line.startswith("* "):
            story.append(Paragraph("• " + _md_inline(line[2:]), styles["ReportBody"]))
        else:
            story.append(Paragraph(_md_inline(line), styles["ReportBody"]))
        i += 1

    _flush_table()
    doc.build(story)
    return out_path


if __name__ == "__main__":
    # Smoke test with canned gathered data (no network needed for the structure;
    # LLM body needs a key, otherwise the fallback body is used).
    sample = (
        "[Step 1 — sec_filing_search] MICROSOFT CORP 10-K filed 2025-07-30\n"
        "[Step 2 — financial_data_api] Ticker MSFT; Revenue 2025: 281,724,000,000; "
        "profit_margin: 39.3%; return_on_equity: 34.0%; trailing_pe: 25.4\n"
        "[Step 3 — web_search] News summary: Azure surpassed $75B annual revenue; "
        "heavy AI investment."
    )
    result = generate_report(
        "Give me a profile of Microsoft", sample, ticker="MSFT",
        sources=["SEC EDGAR 10-K (2025-07-30)", "yfinance", "Tavily web search"],
    )
    print("Report written:")
    print(" ", result)