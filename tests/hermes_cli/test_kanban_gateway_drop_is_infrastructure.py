"""A worker killed by a GATEWAY restart must not be charged to the card.

2026-10-29: review card ``t_74e17583`` ran, the operator restarted the gateway, and the
worker's owner refused the turn — ``Kanban owner refused: Gateway disconnected; turn
outcome is unknown``. The dispatcher booked that as a card crash, incremented
``consecutive_failures`` and consumed the card's single retry (``max_retries=1``); a second
restart exhausted it and the review never produced a verdict.

A gateway drop is a host event. It must reclaim the card WITHOUT increasing the failure
counter — the same treatment a spawn refusal already gets (``infrastructure=True``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _claimed_task(conn) -> str:
    tid = kb.create_task(conn, title="review", assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid


def _dead(error_text: str) -> kbd._DeadWorker:
    return kbd._DeadWorker(
        kind="crash", code=1, error_text=error_text,
        event_kind="crashed", event_payload={},
    )


def test_gateway_drop_does_not_consume_the_retry_budget(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _claimed_task(conn)
        dead = _dead(
            "pid 123 exited with code 1 Worker's last output: 'Kanban owner refused: "
            "Gateway disconnected; turn outcome is unknown.'"
        )

        blocked = kbd._account_crashes(conn, [(tid, 123, "worker", dead)])

        assert blocked == [], "a gateway drop must not trip the breaker"
        row = conn.execute(
            "SELECT consecutive_failures, status FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        assert row["consecutive_failures"] == 0, (
            "a gateway drop charged the card's retry budget"
        )
        assert row["status"] != "blocked"


def test_a_real_crash_is_still_charged(kanban_home: Path) -> None:
    """Control: an ordinary crash must keep incrementing the counter."""
    with kbc.connect_closing() as conn:
        tid = _claimed_task(conn)
        dead = _dead("pid 123 exited with code 1 Worker's last output: 'boom'")

        kbd._account_crashes(conn, [(tid, 123, "worker", dead)])

        row = conn.execute(
            "SELECT consecutive_failures FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        assert row["consecutive_failures"] == 1
