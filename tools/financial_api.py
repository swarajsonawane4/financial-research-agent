"""Financial data tool — structured financials via the yfinance library.

yfinance is a free, no-API-key wrapper around Yahoo Finance. It gives us income
statements, balance sheets, cash-flow statements, and key ratios. The reference
material treats structured financial data as a Tier-2 source (below SEC filings
but above news), so this is the agent's go-to for numbers.

Per the reference document's guidance, financial statements are returned as
structured data (dicts of numbers keyed by period), NOT as free text — numerical
data should be retrieved by field, not by semantic search.

The high-level entry point is `financial_data_api`, matching the registry schema.
"""

from __future__ import annotations

from typing import Optional

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


def _require_yf() -> Optional[dict]:
    """Return an error dict if yfinance isn't installed, else None."""
    if yf is None:
        return {
            "ok": False,
            "error": "yfinance is not installed. Run: pip install yfinance",
        }
    return None


def _frame_to_records(df, years: int) -> dict:
    """Convert a yfinance statement DataFrame to a clean {period: {line: value}} dict.

    yfinance returns statements as DataFrames with line items as rows and period
    end-dates as columns. We transpose so each period is a record, trim to the
    requested number of years, and stringify dates / round numbers for clean JSON.
    """
    if df is None or df.empty:
        return {}

    # columns are timestamps (period end dates), newest first
    out: dict = {}
    for col in list(df.columns)[:years]:
        period = str(col.date()) if hasattr(col, "date") else str(col)
        record = {}
        for line_item, value in df[col].items():
            try:
                # most values are large floats; keep them as ints where sensible
                record[str(line_item)] = None if value != value else float(value)  # NaN check
            except (TypeError, ValueError):
                record[str(line_item)] = None
        out[period] = record
    return out


def financial_data_api(
    ticker: str,
    statement_type: str = "all",
    period: str = "annual",
    years: int = 4,
) -> dict:
    """Retrieve structured financial statements for a company.

    Args:
        ticker: Stock ticker, e.g. "MSFT".
        statement_type: One of "income", "balance", "cash_flow", "ratios", "all".
        period: "annual" or "quarterly".
        years: How many periods of history to return.

    Returns:
        dict with ok flag, ticker, period, and the requested statement data.
        For "ratios", returns a snapshot dict of common valuation/profitability
        ratios from Yahoo's info endpoint.
    """
    err = _require_yf()
    if err:
        return err

    ticker = ticker.strip().upper()
    quarterly = period == "quarterly"

    try:
        tk = yf.Ticker(ticker)

        result: dict = {"ok": True, "ticker": ticker, "period": period}

        want = {"income", "balance", "cash_flow", "ratios", "all"}
        if statement_type not in want:
            return {
                "ok": False,
                "error": f"statement_type must be one of {sorted(want)}.",
            }

        if statement_type in ("income", "all"):
            df = tk.quarterly_income_stmt if quarterly else tk.income_stmt
            result["income_statement"] = _frame_to_records(df, years)

        if statement_type in ("balance", "all"):
            df = tk.quarterly_balance_sheet if quarterly else tk.balance_sheet
            result["balance_sheet"] = _frame_to_records(df, years)

        if statement_type in ("cash_flow", "all"):
            df = tk.quarterly_cashflow if quarterly else tk.cashflow
            result["cash_flow"] = _frame_to_records(df, years)

        if statement_type in ("ratios", "all"):
            info = tk.info or {}
            result["ratios"] = {
                "market_cap": info.get("marketCap"),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "profit_margin": info.get("profitMargins"),
                "return_on_equity": info.get("returnOnEquity"),
                "revenue_growth": info.get("revenueGrowth"),
                "debt_to_equity": info.get("debtToEquity"),
            }

        # If we asked for statements but got nothing back, flag it (likely a bad ticker).
        if statement_type != "ratios":
            non_empty = any(
                result.get(k)
                for k in ("income_statement", "balance_sheet", "cash_flow")
            )
            if not non_empty:
                return {
                    "ok": False,
                    "error": f"No financial data returned for '{ticker}'. "
                    "Check the ticker symbol.",
                }

        return result

    except Exception as exc:  # noqa: BLE001 - refined by error handler on Day 9
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    import json

    print(json.dumps(financial_data_api("MSFT", "income", years=3), indent=2, default=str))