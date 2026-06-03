"""Tool registry — the central catalog the agent uses to discover and call tools.

This is built around STRUCTURED function-calling schemas (the OpenAI / Anthropic
tool-use format), not free-text "Action: tool(...)" parsing. That makes tool
selection far more reliable than the string-parsing approach shown in the
reference material: the LLM emits clean JSON, we validate it against the schema,
then dispatch. Bad calls are caught before they execute.

The registry handles:
  * tool metadata (name, description, JSON schema) for injection into the prompt
  * input validation against each tool's schema
  * dispatch to the underlying Python function
  * per-tool fallback chains (used by the error handler on Day 9)
  * a record of which tools were "useful" (results cited) for the AB-1 metric

The reference document lists 12 tools; the registry is designed so tools can be
added or unlocked at runtime without code changes (the "progression unlock"
mechanic in the brief).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# --- Tool definition ---------------------------------------------------------

@dataclass
class Tool:
    """A single tool: its metadata, schema, implementation, and fallbacks."""

    name: str
    description: str
    parameters: dict                      # JSON Schema for inputs
    fn: Callable[..., dict]               # the implementation
    fallbacks: list[str] = field(default_factory=list)  # names of fallback tools
    tier: int = 99                        # source-reliability tier (lower = more reliable)

    def to_schema(self) -> dict:
        """Render this tool in OpenAI/Anthropic function-calling format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# --- Lightweight JSON-schema validation --------------------------------------

def _validate(args: dict, schema: dict) -> Optional[str]:
    """Validate args against a (simple) JSON schema. Returns an error string or None.

    Supports the subset we actually use: type checks, required fields, and enums.
    Kept dependency-free on purpose; swap for `jsonschema` if you want stricter checks.
    """
    props = schema.get("properties", {})
    required = schema.get("required", [])

    for key in required:
        if key not in args:
            return f"Missing required parameter: '{key}'."

    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    for key, value in args.items():
        if key not in props:
            continue  # ignore unknown extras rather than hard-failing
        spec = props[key]
        expected = spec.get("type")
        if expected and expected in type_map:
            if not isinstance(value, type_map[expected]):
                return (
                    f"Parameter '{key}' should be {expected}, "
                    f"got {type(value).__name__}."
                )
        if "enum" in spec and value not in spec["enum"]:
            return f"Parameter '{key}'='{value}' not in allowed {spec['enum']}."

    return None


# --- The registry ------------------------------------------------------------

class ToolRegistry:
    """Catalog + execution layer for all agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        # call-efficiency tracking for the AB-1 (Tool Efficiency) metric
        self.calls_total = 0
        self.calls_useful = 0
        # memory-hit tracking for the AB-4 (Memory Utilization) metric
        self.memory_hits = 0
        self.external_api_calls = 0

    # -- registration --
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unlock(self, tool: Tool) -> None:
        """Alias for register, used by the progression-unlock mechanic."""
        self.register(tool)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    # -- prompt injection --
    def schemas(self) -> list[dict]:
        """All tool schemas, for injection into the LLM system prompt."""
        return [t.to_schema() for t in self._tools.values()]

    def describe(self) -> str:
        """Detailed tool list for the LLM prompt.

        Includes each parameter's type, whether it's required, and — crucially —
        its allowed values (enum) when constrained. Without the enum, the LLM
        guesses at valid values; feeding it the real constraints reduces errors.
        """
        lines = []
        for t in self._tools.values():
            props = t.parameters.get("properties", {})
            required = set(t.parameters.get("required", []))
            param_parts = []
            for pname, spec in props.items():
                desc = f"{pname}: {spec.get('type', 'any')}"
                if pname in required:
                    desc += ", required"
                if "enum" in spec:
                    desc += f", one of {spec['enum']}"
                param_parts.append(f"    - {desc}")
            params_block = "\n".join(param_parts) if param_parts else "    (no parameters)"
            lines.append(f"- {t.name}: {t.description}\n  params:\n{params_block}")
        return "\n".join(lines)

    # -- execution --
    def call(self, name: str, args: dict, *, count: bool = True) -> dict:
        """Validate and execute a tool call.

        Returns the tool's dict result, or an error dict {"ok": False, "error": ...}.
        Does NOT do fallback/retry here — that's the error handler's job (Day 9).
        This layer just validates, dispatches, and records metrics.
        """
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"Unknown tool '{name}'."}

        err = _validate(args, tool.parameters)
        if err:
            return {"ok": False, "error": f"Invalid arguments for {name}: {err}"}

        if count:
            self.calls_total += 1
            # track external vs memory calls for AB-4
            if name in ("vector_db_search",):
                self.memory_hits += 1
            elif name in ("vector_db_store",):
                pass  # writes aren't "hits" or external lookups
            else:
                self.external_api_calls += 1

        started = time.time()
        try:
            result = tool.fn(**args)
        except Exception as exc:  # noqa: BLE001 - refined by error handler on Day 9
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result.setdefault("_elapsed_s", round(time.time() - started, 3))
        return result

    def mark_useful(self) -> None:
        """Record that the last call's result was actually used in the report (AB-1)."""
        self.calls_useful += 1

    # -- metrics --
    def tool_efficiency(self) -> float:
        """AB-1: ratio of useful tool calls to total tool calls (target >= 0.70)."""
        return self.calls_useful / self.calls_total if self.calls_total else 0.0

    def memory_utilization(self) -> float:
        """AB-4: ratio of memory hits to external API calls (target >= 0.30).

        NOTE: The reference document defines AB-4 as a RATIO but then says to
        compute it as 'memory_hits MULTIPLIED BY total_api_calls'. That is one of
        the planted errors — a ratio is division, not multiplication. Implemented
        correctly here as division. (See ERROR_LOG.md.)
        """
        return self.memory_hits / self.external_api_calls if self.external_api_calls else 0.0


# --- Source-reliability hierarchy (CORRECTED) --------------------------------

# The reference document's Tier list is broken: it ranks anonymous social media
# / forums (Tier 4) as MORE reliable than major news outlets like Reuters,
# Bloomberg, and the FT (Tier 5). That inverts reality and is one of the planted
# errors. The corrected, sensible hierarchy is below (lower number = more
# trustworthy), and it's what the synthesis engine (Day 8) uses for conflict
# resolution. (See ERROR_LOG.md.)
SOURCE_TIERS = {
    "sec_filing": 1,        # legally mandated, audited
    "financial_data_api": 2,  # curated from primary sources
    "major_news": 3,        # Reuters / Bloomberg / FT — professional journalism
    "earnings_call": 4,     # direct management commentary, but subject to spin
    "social_forum": 5,      # crowd-sourced, unverified
}


def tier_of(source_type: str) -> int:
    """Return the reliability tier for a source type (lower = more reliable)."""
    return SOURCE_TIERS.get(source_type, 99)
