"""Function-calling schemas for all 12 tools.

These follow the input specifications in the reference document (Section A2.2)
but are expressed as proper JSON Schema objects (OpenAI / Anthropic tool-use
format) rather than the free-text descriptions in the source. Each schema is
paired with its implementation in agent setup.

Keeping schemas in one place makes them easy to inject into the system prompt
and easy to validate against.
"""

# --- Data-source tools -------------------------------------------------------

SEC_FILING_SEARCH = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL, MSFT."},
        "filing_type": {
            "type": "string",
            "enum": ["10-K", "10-Q", "8-K", "DEF 14A"],
            "description": "Type of SEC filing to retrieve.",
        },
        "year": {"type": "integer", "description": "Filing year (defaults to most recent)."},
    },
    "required": ["ticker", "filing_type"],
}

WEB_SEARCH = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query."},
        "num_results": {"type": "integer", "description": "How many results (default 10)."},
        "date_range": {"type": "string", "description": "Optional recency filter, e.g. 'past_month'."},
    },
    "required": ["query"],
}

EARNINGS_TRANSCRIPT = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string", "description": "Stock ticker."},
        "quarter": {"type": "string", "enum": ["Q1", "Q2", "Q3", "Q4"], "description": "Fiscal quarter."},
        "year": {"type": "integer", "description": "Fiscal year."},
    },
    "required": ["ticker", "quarter", "year"],
}

FINANCIAL_DATA_API = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string", "description": "Stock ticker."},
        "statement_type": {
            "type": "string",
            "enum": ["income", "balance", "cash_flow", "ratios", "all"],
            "description": "Which financial statement(s) to retrieve.",
        },
        "period": {"type": "string", "enum": ["annual", "quarterly"], "description": "Reporting period."},
        "years": {"type": "integer", "description": "How many years of history."},
    },
    "required": ["ticker", "statement_type"],
}

NEWS_SENTIMENT = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Company or topic to analyze."},
        "num_articles": {"type": "integer", "description": "How many articles to sample."},
        "lookback_days": {"type": "integer", "description": "How far back to look."},
    },
    "required": ["query"],
}

COMPANY_PROFILE = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string", "description": "Stock ticker."},
    },
    "required": ["ticker"],
}

PEER_COMPARISON = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string", "description": "Stock ticker of the focal company."},
        "peers": {
            "type": "array",
            "description": (
                "List of peer/competitor stock tickers to compare against, e.g. "
                "['SNOW','DDOG','MDB'] for a data-software company. YOU must "
                "supply these from your knowledge of the company's competitors — "
                "the tool does not auto-discover peers."
            ),
        },
        "num_peers": {"type": "integer", "description": "Max number of peers to include (default 3)."},
        "metrics": {
            "type": "array",
            "description": "Optional subset of metrics, e.g. ['pe_ratio','revenue_growth'].",
        },
    },
    "required": ["ticker", "peers"],
}

# --- Memory tools ------------------------------------------------------------

VECTOR_DB_SEARCH = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Semantic query for long-term memory."},
        "top_k": {"type": "integer", "description": "How many results to return."},
        "filter": {"type": "object", "description": "Optional metadata filter, e.g. {'ticker':'TSLA'}."},
    },
    "required": ["query"],
}

VECTOR_DB_STORE = {
    "type": "object",
    "properties": {
        "content": {"type": "string", "description": "The text/finding to store."},
        "metadata": {
            "type": "object",
            "description": "Metadata: ticker, date, source_type, confidence.",
        },
    },
    "required": ["content", "metadata"],
}

# --- Analysis / output tools -------------------------------------------------

REPORT_GENERATOR = {
    "type": "object",
    "properties": {
        "template": {"type": "string", "description": "Report template name, e.g. 'company_profile'."},
        "sections": {"type": "object", "description": "Section name -> content mapping."},
        "sources": {"type": "array", "description": "List of sources used, for citations."},
    },
    "required": ["template", "sections"],
}

FACT_CHECKER = {
    "type": "object",
    "properties": {
        "claim": {"type": "string", "description": "The specific claim to verify."},
        "sources": {"type": "array", "description": "Optional list of sources to check against."},
    },
    "required": ["claim"],
}

CALCULATION_ENGINE = {
    "type": "object",
    "properties": {
        "calculation_type": {
            "type": "string",
            "enum": ["dcf", "pe_ratio", "growth_rate", "cagr", "margin", "roe", "ev_ebitda"],
            "description": "Which financial calculation to perform.",
        },
        "inputs": {"type": "object", "description": "The numeric inputs for the calculation."},
    },
    "required": ["calculation_type", "inputs"],
}


# Convenience map: tool name -> schema
ALL_SCHEMAS = {
    "sec_filing_search": SEC_FILING_SEARCH,
    "web_search": WEB_SEARCH,
    "earnings_transcript": EARNINGS_TRANSCRIPT,
    "financial_data_api": FINANCIAL_DATA_API,
    "news_sentiment": NEWS_SENTIMENT,
    "company_profile": COMPANY_PROFILE,
    "peer_comparison": PEER_COMPARISON,
    "vector_db_search": VECTOR_DB_SEARCH,
    "vector_db_store": VECTOR_DB_STORE,
    "report_generator": REPORT_GENERATOR,
    "fact_checker": FACT_CHECKER,
    "calculation_engine": CALCULATION_ENGINE,
}