"""An EXTERNAL block must terminate the live worker it orphans.

History (red7 ``t_bbb998f0``, 2026-10-29): ``kanban block`` on a *running* card
cleared ``claim_lock``/``worker_pid`` and closed the run, but never signalled the
process. ``reclaim_task`` DOES signal — but it refuses once the row is no longer
``running``, so the block left a worker writing files while the card read
``blocked`` and every correction path refused it.

Both sides are pinned here:

* external block (no owning run asserted) → the worker must die;
* a worker blocking its OWN card (``expected_run_id`` == its run) → must NOT be
  signalled, or a legitimate self-block kills the caller mid-tool-call.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
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


def _running_task_with_live_worker(conn):
    """Claim a task and attach a REAL live child process as its worker."""
    tid = kb.create_task(conn, title="t", assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        fingerprint = kbd._process_fingerprint(child.pid)
        assert fingerprint is not None, "could not fingerprint the child process"
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_pid = ?, worker_started_at = ?, claim_lock = ? WHERE id = ?",
                (child.pid, fingerprint, f"{kb._host_prefix()}:{child.pid}", tid),
            )
        return tid, claimed, child
    except Exception:
        child.kill()
        child.wait(timeout=10)
        raise


def _worker_terminated_events(conn, tid):
    return [e for e in kb.list_events(conn, tid) if e.kind == "worker_terminated"]


def test_external_block_terminates_the_live_worker(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid, _claimed, child = _running_task_with_live_worker(conn)

        assert kb.block_task(conn, tid, reason="operator park", kind="needs_input")

        # The block must not leave the process behind: it has to die (reaped).
        assert child.wait(timeout=10) is not None, "blocked card left its worker running"

        task = kb.get_task(conn, tid)
        assert task is not None and task.status == "blocked"

        events = _worker_terminated_events(conn, tid)
        assert events, "no worker_terminated event recorded on the blocked card"
        payload = events[-1].payload or {}
        assert payload.get("pid") == child.pid
        assert payload.get("terminated") is True


def test_self_block_does_not_signal_the_owning_worker(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid, claimed, child = _running_task_with_live_worker(conn)
        run_id = claimed.current_run_id

        assert kb.block_task(
            conn, tid, reason="worker self-block", kind="needs_input", expected_run_id=run_id,
        )

        time.sleep(1.0)
        assert child.poll() is None, "self-block killed the worker that called it"
        assert not _worker_terminated_events(conn, tid)

        child.kill()
        child.wait(timeout=10)


def test_termination_failure_still_fires_the_blocked_hook(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The block commits before the signal: a failed kill must not swallow the hook.

    Otherwise the card reads ``blocked`` with no ``kanban_task_blocked`` wake, and
    the operator never learns it needs attention.
    """
    fired: list[str] = []
    monkeypatch.setattr(kb, "_fire_task_hook", lambda event, *a, **kw: fired.append(event))

    def _boom(*_a, **_kw):
        raise RuntimeError("signal path exploded")

    monkeypatch.setattr(kb, "_terminate_orphaned_block_worker", _boom)

    with kbc.connect_closing() as conn:
        tid, _claimed, child = _running_task_with_live_worker(conn)
        try:
            assert kb.block_task(conn, tid, reason="operator park", kind="needs_input")
            task = kb.get_task(conn, tid)
            assert task is not None and task.status == "blocked"
            assert "kanban_task_blocked" in fired, "blocked hook was swallowed by the kill failure"
            assert not _worker_terminated_events(conn, tid)
        finally:
            child.kill()
            child.wait(timeout=10)


def test_external_block_never_signals_its_own_pid(kanban_home: Path) -> None:
    """A self-pid worker row (dashboard/CLI acting as its own worker) is not signalled."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        fingerprint = kbd._process_fingerprint(os.getpid())
        assert fingerprint is not None
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_pid = ?, worker_started_at = ?, claim_lock = ? WHERE id = ?",
                (os.getpid(), fingerprint, f"{kb._host_prefix()}:{os.getpid()}", tid),
            )

        assert kb.block_task(conn, tid, reason="operator park", kind="needs_input")

        assert kb.get_task(conn, tid).status == "blocked"
        assert not _worker_terminated_events(conn, tid), "block signalled the process running the block"
