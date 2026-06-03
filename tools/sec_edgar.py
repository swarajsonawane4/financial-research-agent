"""SEC EDGAR tool — retrieves filings from the SEC's free public APIs.

No API key required. The SEC asks only for a descriptive User-Agent header
identifying your application (with contact info), and a rate limit of <=10
requests/second. We stay well under that.

Two public endpoints are used:
  1. https://www.sec.gov/files/company_tickers.json
     -> maps ticker symbols to CIK numbers (downloaded once, cached on disk).
  2. https://data.sec.gov/submissions/CIK##########.json
     -> the filing history for a company (filing types, dates, accession numbers).

This module exposes a single high-level function, `sec_filing_search`, which is
the form the agent's tool registry will wrap later.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional

import httpx

# --- Configuration -----------------------------------------------------------

# The SEC requires a User-Agent that identifies you with real contact info.

USER_AGENT = "Swaraj Sonawane financial-research-agent (swarajsonawane4@gmail.com)"

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# Where we cache the ticker->CIK map so we don't re-download it every run.
CACHE_DIR = Path.home() / ".cache" / "financial-research-agent"
TICKER_CACHE = CACHE_DIR / "company_tickers.json"
TICKER_CACHE_TTL_SECONDS = 60 * 60 * 24 * 7  # refresh weekly

VALID_FILING_TYPES = {"10-K", "10-Q", "8-K", "DEF 14A"}

_HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}


# --- Ticker -> CIK resolution ------------------------------------------------

def _load_ticker_map() -> dict:
    """Return the SEC's ticker->CIK map, downloading and caching if needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    fresh = (
        TICKER_CACHE.exists()
        and (time.time() - TICKER_CACHE.stat().st_mtime) < TICKER_CACHE_TTL_SECONDS
    )
    if fresh:
        return json.loads(TICKER_CACHE.read_text())

    resp = httpx.get(TICKER_MAP_URL, headers=_HEADERS, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    TICKER_CACHE.write_text(json.dumps(data))
    return data


def ticker_to_cik(ticker: str) -> Optional[int]:
    """Resolve a ticker symbol (e.g. 'MSFT') to its integer CIK, or None."""
    ticker = ticker.strip().upper()
    data = _load_ticker_map()
    # The file is a dict of {"0": {"cik_str": 789019, "ticker": "MSFT", ...}, ...}
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker:
            return int(entry["cik_str"])
    return None


# --- Filing retrieval --------------------------------------------------------

def _get_submissions(cik: int) -> dict:
    """Fetch the full filing history JSON for a CIK."""
    url = SUBMISSIONS_URL.format(cik=cik)
    resp = httpx.get(url, headers=_HEADERS, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def sec_filing_search(
    ticker: str,
    filing_type: str = "10-K",
    year: Optional[int] = None,
) -> dict:
    """Retrieve the most recent matching SEC filing for a company.

    Args:
        ticker: Stock ticker symbol, e.g. "MSFT", "AAPL".
        filing_type: One of "10-K", "10-Q", "8-K", "DEF 14A".
        year: Optional filing year. If given, returns the matching filing from
              that year; otherwise returns the most recent of that type.

    Returns:
        A dict with keys:
          ok            (bool)  whether a filing was found
          ticker        (str)
          cik           (int)
          company       (str)
          filing_type   (str)
          filing_date   (str, YYYY-MM-DD)
          accession     (str)
          primary_doc   (str)   filename of the primary document
          url           (str)   direct link to the filing index
          error         (str)   present only when ok is False
    """
    filing_type = filing_type.strip().upper()
    if filing_type not in VALID_FILING_TYPES:
        return {
            "ok": False,
            "error": f"Unsupported filing_type '{filing_type}'. "
            f"Use one of: {', '.join(sorted(VALID_FILING_TYPES))}.",
        }

    cik = ticker_to_cik(ticker)
    if cik is None:
        return {"ok": False, "error": f"Could not resolve ticker '{ticker}' to a CIK."}

    submissions = _get_submissions(cik)
    company = submissions.get("name", "")
    recent = submissions.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    # Walk filings newest-first; pick the first matching the type (and year).
    for form, date, accession, doc in zip(forms, dates, accessions, primary_docs):
        if form.upper() != filing_type:
            continue
        if year is not None and not date.startswith(str(year)):
            continue

        accession_nodashes = accession.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{accession_nodashes}/{doc}"
        )
        return {
            "ok": True,
            "ticker": ticker.upper(),
            "cik": cik,
            "company": company,
            "filing_type": filing_type,
            "filing_date": date,
            "accession": accession,
            "primary_doc": doc,
            "url": url,
        }

    span = f" for year {year}" if year else ""
    return {
        "ok": False,
        "error": f"No {filing_type} filing found for {ticker.upper()}{span}.",
    }


# --- Manual smoke test -------------------------------------------------------

if __name__ == "__main__":
    result = sec_filing_search("MSFT", "10-K")
    print(json.dumps(result, indent=2))
