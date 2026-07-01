"""Terminal tool — run a shell command with a timeout and bounded output.

Registered as ``terminal`` in the ``terminal`` toolset. The executor calls it
via the registry; it never imports this module directly.

Safety posture for v2.0: this runs arbitrary shell commands in the current
working directory. That is intentional — the whole point of the apprentice
agent is to execute real tasks — but it means the tool must never hang the
loop or flood the context. So:
- Commands run under a hard timeout (default 30s); a timeout is reported as a
  normal (non-crashing) result so the model can react.
- Combined stdout+stderr is captured and truncated to a byte budget, with a
  clear marker when truncated, so a runaway command can't blow the context.

Sandboxing / command allowlisting is a v2.1+ concern (see FUTURE-LEARNINGS).
"""

from __future__ import annotations

import logging
import subprocess

from autodidact.tools.registry import REGISTRY

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 30
_MAX_OUTPUT_BYTES = 16_000  # ~4k tokens; keeps a single tool result bounded


def _truncate(text: str, limit: int = _MAX_OUTPUT_BYTES) -> str:
    """Truncate output to a byte budget, keeping the tail (usually the error).

    Shell failures put the useful message at the end, so when we must drop
    bytes we keep the tail and prepend a marker rather than cutting the head.
    """
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text
    kept = encoded[-limit:].decode("utf-8", errors="replace")
    dropped = len(encoded) - limit
    return f"[... {dropped} bytes truncated ...]\n{kept}"


def run_terminal(args: dict) -> dict:
    """Execute a shell command and return its outcome.

    Args (from the schema):
        command: the shell command line to run.
        timeout: optional per-call timeout in seconds (default 30).

    Returns a dict with ``exit_code``, ``output`` (combined stdout+stderr,
    truncated), and ``timed_out``. A non-zero exit is a normal result, not an
    error — the model reads ``exit_code`` and decides what to do next.
    """
    command = args.get("command")
    if not command or not isinstance(command, str):
        raise ValueError("terminal requires a non-empty 'command' string")

    timeout = args.get("timeout") or _DEFAULT_TIMEOUT_SECONDS

    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        partial = (e.stdout or "") + (e.stderr or "")
        return {
            "exit_code": None,
            "output": _truncate(partial),
            "timed_out": True,
        }

    combined = (completed.stdout or "") + (completed.stderr or "")
    return {
        "exit_code": completed.returncode,
        "output": _truncate(combined),
        "timed_out": False,
    }


REGISTRY.register(
    "terminal",
    description=(
        "Run a shell command in the current working directory and return its "
        "exit code and combined stdout/stderr. Output is truncated if large. "
        "Use for building, testing, git, and inspecting the system."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command line to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Optional timeout in seconds (default 30).",
            },
        },
        "required": ["command"],
    },
    handler=run_terminal,
    toolset="terminal",
)


__all__ = ["run_terminal"]
