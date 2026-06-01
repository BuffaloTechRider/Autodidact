"""Shared rich Console for the wizard modules.

The wizard's interactive prompts (picker, prompts, flow, smoke) all write
to the same Console. Centralizing it here avoids each module owning its
own instance and keeps colors/styling consistent.

cli.py imports the same Console for its own non-wizard output, so the
whole CLI shares one Console object — important for tools like rich's
status spinner that capture output by attribute.
"""

from __future__ import annotations

from rich.console import Console


console = Console()


__all__ = ["console"]
