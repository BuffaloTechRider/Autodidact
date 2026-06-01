"""Generic list-picker used by every interactive wizard prompt.

Two implementations: an arrow-key TUI via ``questionary`` when stdin is a
TTY and the lib is available, falling back to a numbered list with
``typer.prompt`` for CI / piped input.
"""

from __future__ import annotations

import sys

import typer

from autodidact.setup_wizard._console import console


_OTHER_CHOICE = "Other (type a model name)"


def _questionary_available() -> bool:
    """True if questionary can be imported AND stdin is a TTY.

    Falls back to typer.prompt in non-interactive shells (CI, piped input)
    so the wizard works in both modes.
    """
    try:
        import questionary  # noqa: F401
    except ImportError:
        return False
    return sys.stdin.isatty()


def _pick_from_list(title: str, choices: list[str], default: str) -> str:
    """Show a picker for ``choices`` with ``default`` pre-selected.

    Uses questionary.select (arrow-key navigation) when available,
    otherwise prints a numbered list and reads a number from typer.prompt.
    Pressing Enter alone accepts the default; invalid input also returns
    the default rather than looping.
    """
    if _questionary_available():
        import questionary
        # questionary raises on interrupt; keep behavior consistent with typer.
        answer = questionary.select(title, choices=choices, default=default).ask()
        if answer is None:
            # User pressed Ctrl+C; re-raise as KeyboardInterrupt so typer handles it.
            raise KeyboardInterrupt
        return answer

    # Fallback: numbered list.
    console.print(f"[bold]{title}[/bold]")
    for i, choice in enumerate(choices, start=1):
        marker = " (default)" if choice == default else ""
        console.print(f"  {i}. {choice}{marker}")
    default_idx = str(choices.index(default) + 1) if default in choices else "1"
    raw = typer.prompt("Choice", default=default_idx).strip()
    if not raw:
        return default
    # Accept a 1-based index…
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(choices):
            return choices[idx]
    except ValueError:
        pass
    # …or a direct label / substring match against the choice list.
    # This keeps existing integration tests working (they feed provider
    # names like 'openrouter' rather than numbers).
    raw_lower = raw.lower()
    for choice in choices:
        if choice.lower() == raw_lower:
            return choice
    for choice in choices:
        if raw_lower in choice.lower():
            return choice
    return default


__all__ = [
    "_OTHER_CHOICE",
    "_pick_from_list",
    "_questionary_available",
]
