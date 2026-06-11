"""LLM client — a thin, swappable wrapper around the language model.

Isolating the LLM call in one module means the rest of the agent never talks to
Gemini (or any provider) directly. If you want to switch models or providers
later, you change this file and nothing else. The agent just calls
`llm_decide_tool(...)` and gets back a structured decision.

Currently wired to Google Gemini (free tier). Reads GEMINI_API_KEY from the
environment. Defaults to gemini-2.5-flash — the sweet spot for planning quality
vs. free-tier request limits. Override with the GEMINI_MODEL env var.

If the LLM is unavailable (no key, network error, quota exhausted), callers are
expected to fall back to the keyword planner — so the agent degrades gracefully
rather than dying.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Default model: best planning quality within the free tier's request limits.
# Swap to gemini-2.5-flash-lite for more daily requests, or gemini-2.5-pro for
# top reasoning on final runs.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None


def llm_available() -> bool:
    """True if the Gemini library is installed and an API key is configured."""
    return genai is not None and bool(os.getenv("GEMINI_API_KEY"))


def _configure() -> None:
    """Configure the Gemini client with the API key (idempotent)."""
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def llm_decide_tool(query: str, tool_descriptions: str) -> Optional[dict]:
    """Ask the LLM which single tool to call for a query, and with what args.

    This is the Day-4 replacement for the keyword planner. Given the user's
    query and a description of the available tools, the LLM returns a structured
    decision: which tool, what arguments, and a one-line reason.

    Args:
        query: the user's research query.
        tool_descriptions: a text list of available tools and their parameters.

    Returns:
        A dict like:
          {"thought": "...", "tool": "financial_data_api",
           "args": {"ticker": "MSFT", "statement_type": "all"}}
        or None if the LLM is unavailable or the response can't be parsed
        (the caller should then fall back to the keyword planner).
    """
    if not llm_available():
        return None

    _configure()

    system_instruction = (
        "You are the planning brain of an autonomous financial research agent. "
        "Given a user query and a list of available tools, decide which SINGLE "
        "tool to call next and with what arguments. Respond with ONLY a JSON "
        "object (no markdown, no backticks, no prose) in exactly this shape:\n"
        '{"thought": "<one sentence on why>", "tool": "<tool_name>", '
        '"args": {<arguments matching that tool\'s schema>}}\n\n'
        "Rules:\n"
        "- Use a company's ticker symbol (e.g. MSFT, AAPL) in args, not its name.\n"
        "- Pick the tool whose description best matches the query's intent.\n"
        "- If the user asks what they already know / have researched about a "
        "specific company, use vector_db_search AND include a filter with that "
        'company\'s ticker, e.g. {"query": "...", "filter": {"ticker": "TSLA"}}.\n'
        "- Only output the JSON object, nothing else."
    )

    prompt = (
        f"{system_instruction}\n\n"
        f"AVAILABLE TOOLS:\n{tool_descriptions}\n\n"
        f"USER QUERY: {query}\n\n"
        "JSON decision:"
    )

    try:
        model = genai.GenerativeModel(DEFAULT_MODEL)
        response = model.generate_content(prompt)
        text = (response.text or "").strip()

        # Strip accidental markdown code fences if the model adds them.
        if text.startswith("```"):
            text = text.split("```")[1] if "```" in text[3:] else text
            text = text.replace("json", "", 1).strip()
            text = text.strip("`").strip()

        decision = json.loads(text)

        # Basic shape check — must have a tool name and an args dict.
        if "tool" not in decision:
            return None
        decision.setdefault("args", {})
        decision.setdefault("thought", f"Calling {decision['tool']}.")
        return decision

    except Exception:  # noqa: BLE001 - any failure => fall back to keyword planner
        return None


def llm_make_plan(query: str, tool_descriptions: str) -> Optional[list]:
    """Ask the LLM to produce a MULTI-STEP plan for a research query.

    This is the Plan-and-Execute upgrade over single-tool selection. Instead of
    picking one tool, the LLM decomposes the query into an ordered list of tool
    calls — e.g. profiling a company => [filing, financials, news]. The executor
    then runs each step and the synthesizer combines the results.

    Returns a list of step dicts:
        [{"step": 1, "thought": "...", "tool": "...", "args": {...}}, ...]
    or None if the LLM is unavailable / response unparseable (caller falls back
    to single-tool planning).
    """
    if not llm_available():
        return None

    _configure()

    system_instruction = (
        "You are the planning brain of an autonomous financial research agent. "
        "Decompose the user's query into an ordered list of tool calls that, "
        "together, gather everything needed to answer it thoroughly. A broad "
        "query like 'profile company X' usually needs several steps (e.g. its "
        "filing, its financials, and recent news); a narrow query needs just "
        "one. Respond with ONLY a JSON array (no markdown, no prose) of steps in "
        "this exact shape:\n"
        '[{"step": 1, "thought": "<why this step>", "tool": "<tool_name>", '
        '"args": {<args matching that tool\'s schema>}}, ...]\n\n'
        "Rules:\n"
        "- Use ticker symbols (e.g. MSFT) in args, not company names.\n"
        "- Each tool and its args must match the schemas below exactly.\n"
        "- Order steps logically (gather authoritative data before news).\n"
        "- For peer_comparison, supply real competitor tickers in peers=[...] "
        "from your own knowledge, and do NOT pass a metrics filter (the tool "
        "returns its full default metric set).\n"
        "- Use between 1 and 4 steps. Don't pad; only include useful steps.\n"
        "- Do NOT include vector_db_store steps — storage is handled automatically.\n"
        "- Output ONLY the JSON array, nothing else."
    )

    prompt = (
        f"{system_instruction}\n\n"
        f"AVAILABLE TOOLS:\n{tool_descriptions}\n\n"
        f"USER QUERY: {query}\n\n"
        "JSON plan:"
    )

    try:
        model = genai.GenerativeModel(DEFAULT_MODEL)
        response = model.generate_content(prompt)
        text = (response.text or "").strip()

        if text.startswith("```"):
            text = text.split("```")[1] if "```" in text[3:] else text
            text = text.replace("json", "", 1).strip()
            text = text.strip("`").strip()

        plan = json.loads(text)
        if not isinstance(plan, list) or not plan:
            return None

        # Keep only well-formed steps that name a tool.
        clean = []
        for i, step in enumerate(plan, start=1):
            if isinstance(step, dict) and step.get("tool"):
                step.setdefault("step", i)
                step.setdefault("args", {})
                step.setdefault("thought", f"Step {i}: {step['tool']}.")
                clean.append(step)
        return clean or None

    except Exception:  # noqa: BLE001 - any failure => caller falls back
        return None


def llm_synthesize(query: str, gathered: str) -> Optional[str]:
    """Combine multiple tool results into one coherent answer to the query.

    After the executor runs every step in the plan, this takes the collection of
    results and asks the LLM to weave them into a single, readable response that
    actually answers the user's question — the 'synthesize' stage of
    Plan-and-Execute. (A fuller report generator comes later; this is the
    narrative synthesis step.)

    Returns the synthesized text, or None if the LLM is unavailable (caller then
    falls back to showing the raw gathered results).
    """
    if not llm_available():
        return None

    _configure()

    system_instruction = (
        "You are a junior financial analyst. You have gathered data from several "
        "sources to answer a research question. Write a clear, concise, "
        "professional answer based ONLY on the gathered data below. Rules:\n"
        "- Use only facts present in the gathered data; do not invent numbers.\n"
        "- If sources conflict, note it. If data is missing, say so plainly.\n"
        "- Lead with the direct answer, then supporting detail.\n"
        "- Be concise — a few tight paragraphs, not a wall of text.\n"
        "- Plain prose, no preamble like 'Based on the data...'."
    )

    prompt = (
        f"{system_instruction}\n\n"
        f"RESEARCH QUESTION: {query}\n\n"
        f"GATHERED DATA:\n{gathered}\n\n"
        "Your analysis:"
    )

    try:
        model = genai.GenerativeModel(DEFAULT_MODEL)
        response = model.generate_content(prompt)
        return (response.text or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    # Smoke test (needs a valid GEMINI_API_KEY in .env).
    print("LLM available:", llm_available())
    if llm_available():
        tools = (
            "- sec_filing_search(ticker, filing_type): SEC filings\n"
            "- financial_data_api(ticker, statement_type): revenue, margins, ratios\n"
            "- web_search(query): current news\n"
            "- vector_db_search(query): the agent's past research memory"
        )
        for q in ["What is Tesla's profit margin?", "Latest news on NVIDIA?",
                  "Get Apple's 10-K", "What do I already know about Microsoft?"]:
            print(f"\nQuery: {q}")
            print(" ", llm_decide_tool(q, tools))