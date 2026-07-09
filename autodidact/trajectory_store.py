"""Persistence for executor trajectories — resume + learning artifact.

Every ReAct iteration is checkpointed to SQLite as it happens, so a task
interrupted mid-run (crash, kill, budget) can resume from its last committed
step instead of paying for the whole run again — and completed trajectories
stay queryable for skill extraction and threshold tuning (the "beat Hermes"
half of docs/HERMES-LEARNINGS.md §3, #4).

Backs the ``execution_trajectories`` + ``trajectory_steps`` tables. The store
is deliberately thin: it records rows and rebuilds the message list from them;
the executor owns the loop.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from autodidact.llm_client import ChatMessage, ToolCall


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StepRecord:
    """One persisted ReAct iteration."""

    step_index: int
    category: Optional[str]
    tier: Optional[str]
    avg_logprob: Optional[float]
    tool_name: Optional[str]
    tool_arguments: Optional[dict]
    tool_result: Optional[str]
    assistant_content: str


class TrajectoryStore:
    """Reads/writes executor trajectories on a shared sqlite3 connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ── Lifecycle ────────────────────────────────────────────────

    def start(
        self, task: str, *, system: Optional[str], context: Optional[str],
    ) -> str:
        """Create a new 'running' trajectory, returning its id."""
        tid = uuid.uuid4().hex
        now = _now()
        self._conn.execute(
            """INSERT INTO execution_trajectories
               (id, task, system_prompt, context, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'running', ?, ?)""",
            (tid, task, system, context, now, now),
        )
        self._conn.commit()
        return tid

    def record_step(self, trajectory_id: str, step: StepRecord) -> None:
        """Persist one iteration. Committed immediately so resume sees it even
        if the process dies on the next step."""
        self._conn.execute(
            """INSERT OR REPLACE INTO trajectory_steps
               (id, trajectory_id, step_index, category, tier, avg_logprob,
                tool_name, tool_arguments, tool_result, assistant_content, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex, trajectory_id, step.step_index, step.category,
                step.tier, step.avg_logprob, step.tool_name,
                json.dumps(step.tool_arguments) if step.tool_arguments is not None else None,
                step.tool_result, step.assistant_content, _now(),
            ),
        )
        self._conn.execute(
            "UPDATE execution_trajectories SET steps_taken = ?, updated_at = ? WHERE id = ?",
            (step.step_index, _now(), trajectory_id),
        )
        self._conn.commit()

    def finish(
        self, trajectory_id: str, *, status: str, answer: str,
        escalations: int, tools_used: list[str], cost_usd: float,
    ) -> None:
        """Mark a trajectory terminal with its final result."""
        self._conn.execute(
            """UPDATE execution_trajectories
               SET status = ?, answer = ?, escalations = ?, tools_used = ?,
                   cost_usd = ?, updated_at = ?
               WHERE id = ?""",
            (status, answer, escalations, json.dumps(tools_used), cost_usd,
             _now(), trajectory_id),
        )
        self._conn.commit()

    # ── Resume ───────────────────────────────────────────────────

    def load_steps(self, trajectory_id: str) -> list[StepRecord]:
        """Return committed steps in order (empty if none / unknown id)."""
        rows = self._conn.execute(
            """SELECT step_index, category, tier, avg_logprob, tool_name,
                      tool_arguments, tool_result, assistant_content
               FROM trajectory_steps WHERE trajectory_id = ?
               ORDER BY step_index""",
            (trajectory_id,),
        ).fetchall()
        out: list[StepRecord] = []
        for r in rows:
            out.append(StepRecord(
                step_index=r[0], category=r[1], tier=r[2], avg_logprob=r[3],
                tool_name=r[4],
                tool_arguments=json.loads(r[5]) if r[5] else None,
                tool_result=r[6], assistant_content=r[7],
            ))
        return out

    def header(self, trajectory_id: str) -> Optional[dict]:
        """Return task/system/context/status for a trajectory, or None."""
        row = self._conn.execute(
            """SELECT task, system_prompt, context, status
               FROM execution_trajectories WHERE id = ?""",
            (trajectory_id,),
        ).fetchone()
        if row is None:
            return None
        return {"task": row[0], "system": row[1], "context": row[2], "status": row[3]}

    def replay_messages(self, trajectory_id: str) -> list[ChatMessage]:
        """Rebuild the message list from persisted steps: for each recorded
        step, the assistant tool-call turn and its role=tool result. The caller
        prepends the system + initial user turns (rebuilt from the header)."""
        messages: list[ChatMessage] = []
        for s in self.load_steps(trajectory_id):
            if not s.tool_name:
                continue
            call = ToolCall(
                id=f"call_{s.step_index}", name=s.tool_name,
                arguments=s.tool_arguments or {},
            )
            messages.append(ChatMessage(
                role="assistant", content=s.assistant_content, tool_calls=[call],
            ))
            messages.append(ChatMessage(
                role="tool", content=s.tool_result or "",
                tool_call_id=call.id, name=s.tool_name,
            ))
        return messages


__all__ = ["TrajectoryStore", "StepRecord"]
