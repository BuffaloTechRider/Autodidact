"""Tool collection for the v2 execution loop.

Importing this package populates the shared ``REGISTRY`` by importing every
tool module for its registration side effects. The executor imports
``autodidact.tools`` once, then works through ``REGISTRY``.

Adding a tool: create ``tools/<name>.py`` that calls ``REGISTRY.register(...)``
at module level, then import it here so the side effect fires.
"""

from __future__ import annotations

from autodidact.tools.registry import REGISTRY, ToolEntry, ToolRegistry

# Import for registration side effects (order = prompt order).
from autodidact.tools import terminal as _terminal  # noqa: E402,F401
from autodidact.tools import file_ops as _file_ops  # noqa: E402,F401

__all__ = ["REGISTRY", "ToolEntry", "ToolRegistry"]
