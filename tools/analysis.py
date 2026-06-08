"""Calculation engine — deterministic financial math.

A pure-Python tool (no external calls, no API keys) for the quantitative
calculations a financial analyst needs: growth rates, CAGR, margins, simple DCF,
P/E, EV/EBITDA, ROE. Keeping these as an explicit tool — rather than asking the
LLM to do arithmetic in its head — matters: LLMs are unreliable at exact math,
so routing numbers through real code keeps the figures correct and auditable.

Entry point is `calculation_engine`, matching the registry schema.
"""

from __future__ import annotations

from typing import Optional


def _growth_rate(inputs: dict) -> dict:
    """Simple period-over-period growth: (new - old) / old."""
    old = inputs.get("old_value")
    new = inputs.get("new_value")
    if old in (None, 0) or new is None:
        return {"ok": False, "error": "growth_rate needs non-zero 'old_value' and 'new_value'."}
    rate = (new - old) / abs(old)
    return {"ok": True, "calculation": "growth_rate", "result": round(rate, 4),
            "result_pct": f"{rate * 100:.2f}%"}


def _cagr(inputs: dict) -> dict:
    """Compound annual growth rate: (end/begin)^(1/years) - 1."""
    begin = inputs.get("begin_value")
    end = inputs.get("end_value")
    years = inputs.get("years")
    if not begin or not end or not years or begin <= 0 or years <= 0:
        return {"ok": False, "error": "cagr needs positive 'begin_value', 'end_value', 'years'."}
    cagr = (end / begin) ** (1 / years) - 1
    return {"ok": True, "calculation": "cagr", "result": round(cagr, 4),
            "result_pct": f"{cagr * 100:.2f}%"}


def _margin(inputs: dict) -> dict:
    """Margin: numerator / revenue (e.g. net income / revenue)."""
    numerator = inputs.get("numerator")
    revenue = inputs.get("revenue")
    if revenue in (None, 0) or numerator is None:
        return {"ok": False, "error": "margin needs 'numerator' and non-zero 'revenue'."}
    m = numerator / revenue
    return {"ok": True, "calculation": "margin", "result": round(m, 4),
            "result_pct": f"{m * 100:.2f}%"}


def _pe_ratio(inputs: dict) -> dict:
    """Price-to-earnings: price / earnings_per_share."""
    price = inputs.get("price")
    eps = inputs.get("eps")
    if eps in (None, 0) or price is None:
        return {"ok": False, "error": "pe_ratio needs 'price' and non-zero 'eps'."}
    return {"ok": True, "calculation": "pe_ratio", "result": round(price / eps, 2)}


def _roe(inputs: dict) -> dict:
    """Return on equity: net_income / shareholders_equity."""
    ni = inputs.get("net_income")
    equity = inputs.get("shareholders_equity")
    if equity in (None, 0) or ni is None:
        return {"ok": False, "error": "roe needs 'net_income' and non-zero 'shareholders_equity'."}
    r = ni / equity
    return {"ok": True, "calculation": "roe", "result": round(r, 4), "result_pct": f"{r * 100:.2f}%"}


def _ev_ebitda(inputs: dict) -> dict:
    """Enterprise-value to EBITDA multiple."""
    ev = inputs.get("enterprise_value")
    ebitda = inputs.get("ebitda")
    if ebitda in (None, 0) or ev is None:
        return {"ok": False, "error": "ev_ebitda needs 'enterprise_value' and non-zero 'ebitda'."}
    return {"ok": True, "calculation": "ev_ebitda", "result": round(ev / ebitda, 2)}


def _dcf(inputs: dict) -> dict:
    """Simple discounted cash flow valuation.

    Discounts a list of projected free cash flows plus a terminal value back to
    present value. Inputs:
      cash_flows: list of projected FCF, year 1..N
      discount_rate: e.g. 0.10 for 10%
      terminal_growth: perpetuity growth rate for terminal value (e.g. 0.025)
    """
    cfs = inputs.get("cash_flows")
    r = inputs.get("discount_rate")
    g = inputs.get("terminal_growth", 0.025)
    if not cfs or not isinstance(cfs, list) or r in (None, 0):
        return {"ok": False, "error": "dcf needs a 'cash_flows' list and non-zero 'discount_rate'."}
    if r <= g:
        return {"ok": False, "error": "dcf requires discount_rate > terminal_growth."}

    pv = 0.0
    for i, cf in enumerate(cfs, start=1):
        pv += cf / ((1 + r) ** i)
    # Terminal value via Gordon growth on the last projected cash flow.
    terminal = (cfs[-1] * (1 + g)) / (r - g)
    pv_terminal = terminal / ((1 + r) ** len(cfs))
    total = pv + pv_terminal
    return {"ok": True, "calculation": "dcf",
            "result": round(total, 2),
            "pv_explicit": round(pv, 2),
            "pv_terminal": round(pv_terminal, 2)}


_CALCULATIONS = {
    "growth_rate": _growth_rate,
    "cagr": _cagr,
    "margin": _margin,
    "pe_ratio": _pe_ratio,
    "roe": _roe,
    "ev_ebitda": _ev_ebitda,
    "dcf": _dcf,
}


def calculation_engine(calculation_type: str, inputs: Optional[dict] = None) -> dict:
    """Perform a named financial calculation on the given inputs.

    Args:
        calculation_type: one of growth_rate, cagr, margin, pe_ratio, roe,
                           ev_ebitda, dcf.
        inputs: a dict of the numeric inputs that calculation needs.

    Returns a dict with ok flag and the result (plus a percentage form where
    relevant).
    """
    fn = _CALCULATIONS.get(calculation_type)
    if fn is None:
        return {"ok": False,
                "error": f"Unknown calculation '{calculation_type}'. "
                         f"Available: {', '.join(sorted(_CALCULATIONS))}."}
    try:
        return fn(inputs or {})
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    import json
    tests = [
        ("growth_rate", {"old_value": 245122, "new_value": 281724}),
        ("cagr", {"begin_value": 211915, "end_value": 281724, "years": 2}),
        ("margin", {"numerator": 101832, "revenue": 281724}),
        ("pe_ratio", {"price": 450, "eps": 13.64}),
        ("roe", {"net_income": 101832, "shareholders_equity": 343479}),
        ("dcf", {"cash_flows": [70000, 75000, 80000, 85000, 90000],
                 "discount_rate": 0.10, "terminal_growth": 0.03}),
    ]
    for calc, inp in tests:
        print(calc, "->", json.dumps(calculation_engine(calc, inp)))
