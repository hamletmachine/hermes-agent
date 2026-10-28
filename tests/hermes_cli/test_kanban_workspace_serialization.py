"""One workspace = one writer: the dispatcher must serialize by resolved path.

Regression for the 2026-10-27 thread-truth collision: two cards whose resolved
``workspace_path`` was the same worktree were claimed and spawned in the same
second, and two workers wrote one checkout (a half-applied edit left
``NameError`` in the tree).  A task whose resolved workspace is held by another
live claim must stay in ``ready`` with a recorded reason, not spawn.
"""
from __future__ import annotations

import os
from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


def _isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    # The host's live memory pressure must not cap this tick.
    monkeypatch.setattr(kbd, "_system_memory_sample", dict)
    return home


class _Spawn:
    """Records spawn calls and reports no PID (nothing for the reaper to find)."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def __call__(self, task, workspace, board=None):
        self.calls.append((task.id, str(workspace)))

    @property
    def ids(self) -> list[str]:
        return [tid for tid, _ in self.calls]


def _hold_claim(conn, task_id: str):
    """A live claim, exactly as ``claim_task`` leaves it before a spawn."""
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    return claimed


def _release(conn, task_id: str) -> None:
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'done', claim_lock = NULL, claim_expires = NULL "
            "WHERE id = ?",
            (task_id,),
        )


def test_live_claim_on_same_workspace_defers_second_card(tmp_path, monkeypatch):
    """RED: the second card spawned into a workspace a live run already held."""
    _isolated_home(tmp_path, monkeypatch)
    shared = tmp_path / "one-worktree"
    shared.mkdir()
    spawn = _Spawn()
    with kbc.connect() as conn:
        holder = kb.create_task(
            conn, title="holder", assignee="red",
            workspace_kind="dir", workspace_path=str(shared),
        )
        second = kb.create_task(
            conn, title="second", assignee="red",
            workspace_kind="dir", workspace_path=str(shared),
        )
        _hold_claim(conn, holder)
        res = kbd.dispatch_once(conn, spawn_fn=spawn, reconcile_orphans=False)
        deferred = kb.get_task(conn, second)
        events = [(e.kind, e.payload) for e in kb.list_events(conn, second)]
        # A deferral is a WAIT, not a failure: no claim, no failure counted, and
        # the stamped reason must not read as a quota/auth wall to the guard.
        guard = kbd.check_respawn_guard(conn, second)
    assert second not in spawn.ids, f"spawned into a held workspace: {spawn.calls}"
    assert [tid for tid, _w, _p in res.skipped_workspace_busy] == [second]
    assert deferred is not None
    assert deferred.status == "ready", deferred.status
    assert deferred.claim_lock is None
    assert deferred.consecutive_failures == 0
    assert deferred.last_failure_error and "workspace busy" in deferred.last_failure_error
    assert guard is None, guard
    busy = [p or {} for kind, p in events if kind == "workspace_busy"]
    assert busy, [kind for kind, _p in events]
    assert busy[0].get("holder_task_id") == holder
    assert busy[0].get("workspace") == str(shared)

    # Control: the SAME fixture spawns once the holder's claim is gone, so the
    # gate above cannot pass vacuously (it is the claim, not the fixture).
    with kbc.connect() as conn:
        _release(conn, holder)
        res2 = kbd.dispatch_once(conn, spawn_fn=spawn, reconcile_orphans=False)
    assert [tid for tid, _w, _p in res2.spawned] == [second]
    assert spawn.ids == [second]


def test_same_tick_collision_spawns_only_one_worker(tmp_path, monkeypatch):
    """Two ready cards on one workspace: the tick spawns one, defers the other.

    This is the incident shape — both cards ready, one tick, two spawns.
    """
    _isolated_home(tmp_path, monkeypatch)
    shared = tmp_path / "one-worktree"
    shared.mkdir()
    spawn = _Spawn()
    with kbc.connect() as conn:
        first = kb.create_task(
            conn, title="first", assignee="red",
            workspace_kind="dir", workspace_path=str(shared),
        )
        second = kb.create_task(
            conn, title="second", assignee="red",
            workspace_kind="dir", workspace_path=str(shared),
        )
        res = kbd.dispatch_once(conn, spawn_fn=spawn, reconcile_orphans=False)
        states = {}
        for tid in (first, second):
            row = kb.get_task(conn, tid)
            assert row is not None
            states[tid] = row.status
    assert len(spawn.ids) == 1, spawn.calls
    winner = spawn.ids[0]
    loser = second if winner == first else first
    assert [tid for tid, _w, _p in res.skipped_workspace_busy] == [loser]
    assert res.skipped_workspace_busy[0][2] == winner
    assert states[winner] == "running"
    assert states[loser] == "ready"


def test_distinct_workspaces_still_spawn_in_one_tick(tmp_path, monkeypatch):
    """The gate must not serialize unrelated workspaces."""
    _isolated_home(tmp_path, monkeypatch)
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    spawn = _Spawn()
    with kbc.connect() as conn:
        first = kb.create_task(
            conn, title="first", assignee="red",
            workspace_kind="dir", workspace_path=str(left),
        )
        second = kb.create_task(
            conn, title="second", assignee="red",
            workspace_kind="dir", workspace_path=str(right),
        )
        res = kbd.dispatch_once(conn, spawn_fn=spawn, reconcile_orphans=False)
    assert sorted(tid for tid, _w, _p in res.spawned) == sorted([first, second])
    assert sorted(spawn.ids) == sorted([first, second])
    assert res.skipped_workspace_busy == []


def test_equivalent_path_spellings_are_serialized(tmp_path, monkeypatch):
    """A symlink and a ``..`` spelling resolve to the same workspace."""
    _isolated_home(tmp_path, monkeypatch)
    shared = tmp_path / "real-wt"
    shared.mkdir()
    link = tmp_path / "link-wt"
    os.symlink(shared, link)
    spawn = _Spawn()
    with kbc.connect() as conn:
        holder = kb.create_task(
            conn, title="holder", assignee="red",
            workspace_kind="dir", workspace_path=str(shared),
        )
        second = kb.create_task(
            conn, title="second", assignee="red",
            workspace_kind="dir", workspace_path=str(link / ".." / "link-wt"),
        )
        _hold_claim(conn, holder)
        res = kbd.dispatch_once(conn, spawn_fn=spawn, reconcile_orphans=False)
    assert second not in spawn.ids
    assert [tid for tid, _w, _p in res.skipped_workspace_busy] == [second]


def test_busy_deferral_on_auth_shaped_path_still_spawns_after_release(tmp_path, monkeypatch):
    """The stamped deferral reason must not park the card as a quota/auth wall.

    ``check_respawn_guard`` matches quota/auth words in ``last_failure_error``;
    a workspace path containing "auth" (or "429", "billing") must not turn a
    plain wait into an auto-block.
    """
    _isolated_home(tmp_path, monkeypatch)
    shared = tmp_path / "auth-service-worktree"
    shared.mkdir()
    spawn = _Spawn()
    with kbc.connect() as conn:
        holder = kb.create_task(
            conn, title="holder", assignee="red",
            workspace_kind="dir", workspace_path=str(shared),
        )
        second = kb.create_task(
            conn, title="second", assignee="red",
            workspace_kind="dir", workspace_path=str(shared),
        )
        _hold_claim(conn, holder)
        kbd.dispatch_once(conn, spawn_fn=spawn, reconcile_orphans=False)
        deferred = kb.get_task(conn, second)
        guard = kbd.check_respawn_guard(conn, second)
    assert second not in spawn.ids
    assert deferred is not None and deferred.status == "ready"
    assert guard is None, guard
    with kbc.connect() as conn:
        _release(conn, holder)
        res = kbd.dispatch_once(conn, spawn_fn=spawn, reconcile_orphans=False)
    assert [tid for tid, _w, _p in res.spawned] == [second]


def test_busy_deferral_records_one_event_across_ticks(tmp_path, monkeypatch):
    """A held workspace defers every tick; the reason is written once, not per tick."""
    _isolated_home(tmp_path, monkeypatch)
    shared = tmp_path / "one-worktree"
    shared.mkdir()
    spawn = _Spawn()
    with kbc.connect() as conn:
        holder = kb.create_task(
            conn, title="holder", assignee="red",
            workspace_kind="dir", workspace_path=str(shared),
        )
        second = kb.create_task(
            conn, title="second", assignee="red",
            workspace_kind="dir", workspace_path=str(shared),
        )
        _hold_claim(conn, holder)
        for _ in range(3):
            kbd.dispatch_once(conn, spawn_fn=spawn, reconcile_orphans=False)
        kinds = [e.kind for e in kb.list_events(conn, second)]
        still_ready = kb.get_task(conn, second)
    assert second not in spawn.ids
    assert kinds.count("workspace_busy") == 1, kinds
    assert still_ready is not None and still_ready.status == "ready"
