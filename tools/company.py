"""Company profile and peer-comparison tools (via yfinance).

Two tools that round out the fundamental-research toolkit:

  * company_profile  — the qualitative overview: sector, industry, business
                        summary, employee count, country, website. Answers
                        "what does this company actually do?"
  * peer_comparison   — compares a company against named peers on key valuation
                        and profitability metrics, so the agent can say whether
                        a company is cheap or expensive relative to its sector.

Both use yfinance (free, no API key), consistent with financial_api.py.
"""

from __future__ import annotations

from typing import Optional

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


def _need_yf() -> Optional[dict]:
    if yf is None:
        return {"ok": False, "error": "yfinance not installed. Run: pip install yfinance"}
    return None


def company_profile(ticker: str) -> dict:
    """Return a qualitative profile of a company.

    Args:
        ticker: stock ticker, e.g. "MSFT".

    Returns a dict with ok flag plus sector, industry, summary, employees,
    country, website, and the long business description.
    """
    err = _need_yf()
    if err:
        return err

    ticker = ticker.strip().upper()
    try:
        info = yf.Ticker(ticker).info or {}
        if not info.get("longName") and not info.get("shortName"):
            return {"ok": False, "error": f"No profile found for '{ticker}'. Check the ticker."}
        return {
            "ok": True,
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "employees": info.get("fullTimeEmployees"),
            "website": info.get("website"),
            "summary": info.get("longBusinessSummary"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _metric_snapshot(ticker: str) -> dict:
    """Pull the comparison metrics for one ticker (best-effort)."""
    info = yf.Ticker(ticker).info or {}
    return {
        "ticker": ticker,
        "name": info.get("shortName") or ticker,
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "price_to_book": info.get("priceToBook"),
        "profit_margin": info.get("profitMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "return_on_equity": info.get("returnOnEquity"),
    }


def peer_comparison(
    ticker: str,
    peers: Optional[list] = None,
    num_peers: int = 3,
    metrics: Optional[list] = None,
) -> dict:
    """Compare a company against peers on key metrics.

    Args:
        ticker: the focal company's ticker.
        peers: optional explicit list of peer tickers. If omitted, the tool
               returns the focal company's metrics and notes that peers should
               be supplied (auto-discovery of peers is unreliable for free data).
        num_peers: cap on peers to include.
        metrics: optional subset of metric keys to return.

    Returns a dict with ok flag and a `comparison` list of metric snapshots
    (focal company first), plus the metric keys compared.
    """
    err = _need_yf()
    if err:
        return err

    ticker = ticker.strip().upper()
    peer_list = [p.strip().upper() for p in (peers or [])][:num_peers]

    try:
        rows = [_metric_snapshot(ticker)]
        for p in peer_list:
            if p and p != ticker:
                rows.append(_metric_snapshot(p))

        # Optionally narrow to requested metrics (always keep ticker + name).
        if metrics:
            keep = {"ticker", "name", *metrics}
            rows = [{k: v for k, v in row.items() if k in keep} for row in rows]

        result = {"ok": True, "focal": ticker, "comparison": rows}
        if not peer_list:
            result["note"] = ("No peers supplied; returned focal company only. "
                              "Pass peers=['TICKER', ...] to compare.")
        return result
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    import json
    print("=== company_profile(MSFT) ===")
    prof = company_profile("MSFT")
    if prof.get("ok"):
        print(f"  {prof['name']} — {prof['sector']} / {prof['industry']}")
        print(f"  Employees: {prof['employees']}, Country: {prof['country']}")
        print(f"  Summary: {(prof['summary'] or '')[:160]}...")
    else:
        print(prof)
    print("\n=== peer_comparison(MSFT, [GOOGL, AAPL]) ===")
    print(json.dumps(peer_comparison("MSFT", peers=["GOOGL", "AAPL"]), indent=2, default=str))
