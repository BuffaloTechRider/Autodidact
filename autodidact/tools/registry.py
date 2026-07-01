"""Self-registering tool collection for the v2 execution loop.

Each tool module (terminal, file_ops, ...) calls :func:`register` at import
time. The executor asks the registry for OpenAI function-calling schemas to
put in the prompt, then routes tool calls back through :meth:`dispatch`.

Design (from Hermes' self-registering pattern, adapted):
- Tools are plain functions with a typed schema in OpenAI function-calling
  format. The registry never introspects Python signatures — the schema is
  the single source of truth for what the LLM sees.
- Dispatch returns a JSON string, matching the OpenAI tool-result contract
  (the executor appends it verbatim as a ``role="tool"`` message).
- Arguments are coerced to the schema's declared types before the handler
  runs, so a model that emits ``"3"`` for an integer arg doesn't blow up the
  handler. Coercion never raises — an uncoercible value passes through and
  the handler decides what to do.
- Toolsets group tools so a caller can enable a subset (e.g. only ``file``
  tools during a read-only task).

The registry is a module-level singleton (``REGISTRY``). Registration is
idempotent per name: re-registering the same name replaces the entry, which
keeps re-imports (common under pytest) from raising.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# A handler takes the (coerced) argument dict and returns a JSON-serializable
# result. The registry serializes the return value to a string for the caller.
ToolHandler = Callable[[dict], Any]


@dataclass(frozen=True)
class ToolEntry:
    """A single registered tool.

    ``schema`` is the OpenAI function-calling object: ``{"type": "function",
    "function": {"name", "description", "parameters": {JSON Schema}}}``. The
    registry builds this for you in :meth:`ToolRegistry.register` from the
    ``name``/``description``/``parameters`` you pass, so callers don't repeat
    the wrapper boilerplate.
    """

    name: str
    description: str
    schema: dict
    handler: ToolHandler
    toolset: str = "default"


class ToolRegistry:
    """Registry of callable tools, keyed by name.

    Not a global by itself — the module exposes a shared ``REGISTRY`` instance
    that the tool modules populate. Tests can construct their own isolated
    registry to avoid cross-test leakage.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(
        self,
        name: str,
        *,
        description: str,
        parameters: dict,
        handler: ToolHandler,
        toolset: str = "default",
    ) -> None:
        """Register (or replace) a tool.

        ``parameters`` is a JSON Schema object describing the arguments (the
        ``{"type": "object", "properties": {...}, "required": [...]}`` block).
        The full OpenAI function wrapper is built and stored in the entry.
        """
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        if name in self._tools:
            logger.debug("Re-registering tool %r (replacing existing entry)", name)
        self._tools[name] = ToolEntry(
            name=name,
            description=description,
            schema=schema,
            handler=handler,
            toolset=toolset,
        )

    def get_schemas(self, toolsets: Optional[list[str]] = None) -> list[dict]:
        """Return function-calling schemas, optionally filtered by toolset.

        With ``toolsets=None`` every registered tool is returned. Ordering is
        registration order (dict insertion order) for stable prompts.
        """
        entries = self._tools.values()
        if toolsets is not None:
            allowed = set(toolsets)
            entries = [e for e in entries if e.toolset in allowed]
        return [e.schema for e in entries]

    def names(self, toolsets: Optional[list[str]] = None) -> list[str]:
        """Return registered tool names, optionally filtered by toolset."""
        if toolsets is None:
            return list(self._tools.keys())
        allowed = set(toolsets)
        return [n for n, e in self._tools.items() if e.toolset in allowed]

    def dispatch(self, name: str, arguments: dict) -> str:
        """Run a tool by name and return its result as a JSON string.

        The result envelope is always a JSON object so the caller can parse it
        uniformly. On success: ``{"ok": true, "result": <value>}``. On a
        missing tool or handler exception: ``{"ok": false, "error": "..."}``.
        A handler that raises never propagates — the executor should see the
        failure as tool output and decide whether to retry or escalate, not
        crash the loop.
        """
        entry = self._tools.get(name)
        if entry is None:
            return _envelope(ok=False, error=f"unknown tool: {name!r}")

        coerced = _coerce_arguments(entry.schema, arguments)
        try:
            result = entry.handler(coerced)
        except Exception as e:  # noqa: BLE001 — surface as tool output, don't crash the loop
            logger.warning("Tool %r raised: %s", name, e)
            return _envelope(ok=False, error=f"{type(e).__name__}: {e}")
        return _envelope(ok=True, result=result)


def _envelope(*, ok: bool, result: Any = None, error: Optional[str] = None) -> str:
    """Serialize a tool result envelope to a JSON string.

    Non-JSON-serializable results are stringified rather than raising, so an
    accidental return of a Path or bytes still produces valid tool output.
    """
    payload: dict[str, Any] = {"ok": ok}
    if ok:
        payload["result"] = result
    else:
        payload["error"] = error
    return json.dumps(payload, default=str)


def _coerce_arguments(schema: dict, arguments: dict) -> dict:
    """Coerce argument values to the JSON-Schema-declared types.

    Fixes the common LLM mistake of emitting ``"3"``/``"true"`` for
    integer/boolean args. Only top-level properties are coerced, and only when
    the target type is unambiguous. An uncoercible value is left untouched for
    the handler to reject.
    """
    if not isinstance(arguments, dict):
        return {}
    props = (schema.get("function", {}).get("parameters", {}).get("properties", {}))
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        target = props.get(key, {}).get("type") if isinstance(props.get(key), dict) else None
        out[key] = _coerce_value(value, target)
    return out


def _coerce_value(value: Any, target_type: Optional[str]) -> Any:
    """Best-effort coerce a single value to a JSON Schema scalar type."""
    if target_type is None or value is None:
        return value
    try:
        if target_type == "integer" and not isinstance(value, bool):
            if isinstance(value, str) and value.strip().lstrip("-").isdigit():
                return int(value)
            if isinstance(value, float) and value.is_integer():
                return int(value)
        elif target_type == "number" and not isinstance(value, bool):
            if isinstance(value, str):
                return float(value)
        elif target_type == "boolean":
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("true", "1", "yes"):
                    return True
                if lowered in ("false", "0", "no"):
                    return False
    except (ValueError, TypeError):
        return value
    return value


# Shared singleton populated by the tool modules at import time.
REGISTRY = ToolRegistry()


__all__ = ["ToolEntry", "ToolRegistry", "REGISTRY"]
