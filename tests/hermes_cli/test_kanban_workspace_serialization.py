"""One workspace = one writer: creation inherits a shared dir, dispatch serializes it.

Regression for the 2026-10-27 thread-truth collision.  RETRACTION of the first
draft of this docstring: the colliding pair did **not** store the same
``workspace_path``.  Board rows show review card ``t_7fb5b97c`` held
``workspace_kind='dir'`` / ``/home/island/ideas/red7/.worktrees/thread-truth``,
while duplicate FIX card ``t_b4ecddaf`` was created ``scratch`` with
``workspace_path=NULL`` (its ``created`` event carries
``creator_task_id=t_7fb5b97c``); run 209's own task prompt reads
``Workspace: scratch @ .../workspaces/t_b4ecddaf`` and its tool calls
``cd``-ed into ``.../thread-truth`` (msg 59515) and patched absolute paths
there (msgs 59563/59588/59607).  A stored-path comparison alone would not
have deferred it.

Two mechanisms close the class: ``create_task`` inherits a ``dir`` creator's
path for a card that names no workspace (``workspace_inherited_from`` on the
``created`` event), and the dispatch gate defers any card whose RESOLVED
workspace is held by another live claim.  A held card stays in its lane with a
recorded reason and spawns after release; it is never a failure.
"""
from __future__ import annotations

import os
from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_workspace as kbw


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


def test_child_created_without_workspace_inherits_creator_dir_and_defers(tmp_path, monkeypatch):
    """The real incident shape: the duplicate card named no workspace at all.

    ``t_b4ecddaf`` was created from review card ``t_7fb5b97c`` (``dir`` on the
    thread-truth checkout) with no workspace, so it resolved to its own scratch
    dir and the dispatcher could not see that its worker would edit the shared
    checkout.  Inheritance must store the creator's ``dir`` path, defer the
    child while the creator's claim is live, and spawn it after release.
    """
    _isolated_home(tmp_path, monkeypatch)
    shared = tmp_path / "thread-truth"
    shared.mkdir()
    spawn = _Spawn()
    with kbc.connect() as conn:
        creator = kb.create_task(
            conn, title="code review", assignee="red",
            workspace_kind="dir", workspace_path=str(shared),
        )
        child = kb.create_task(
            conn, title="fix", assignee="red", creator_task_id=creator,
            body=f"edit bcn/lib/thread_truth.py under {shared}",
        )
        row = kb.get_task(conn, child)
        created = [e for e in kb.list_events(conn, child) if e.kind == "created"]
        _hold_claim(conn, creator)
        res = kbd.dispatch_once(conn, spawn_fn=spawn, reconcile_orphans=False)
        deferred = kb.get_task(conn, child)
    assert row is not None
    assert (row.workspace_kind, row.workspace_path) == ("dir", str(shared)), (
        "child did not inherit the creator's dir; it would spawn into its own "
        f"scratch dir while its instructions point at {shared}"
    )
    assert created and (created[0].payload or {}).get("workspace_inherited_from") == creator
    assert child not in spawn.ids, spawn.calls
    assert [tid for tid, _w, _p in res.skipped_workspace_busy] == [child]
    assert res.skipped_workspace_busy[0][2] == creator
    assert deferred is not None and deferred.status == "ready"
    assert deferred.claim_lock is None and deferred.consecutive_failures == 0

    # Control: the SAME child spawns once the creator's claim is released.
    with kbc.connect() as conn:
        _release(conn, creator)
        res2 = kbd.dispatch_once(conn, spawn_fn=spawn, reconcile_orphans=False)
    assert [tid for tid, _w, _p in res2.spawned] == [child]
    assert spawn.calls == [(child, str(shared))]


def test_explicit_child_workspace_wins_over_creator_dir(tmp_path, monkeypatch):
    """An explicit workspace (or an explicit scratch opt-out) is never overridden."""
    _isolated_home(tmp_path, monkeypatch)
    shared = tmp_path / "thread-truth"
    other = tmp_path / "somewhere-else"
    shared.mkdir()
    other.mkdir()
    with kbc.connect() as conn:
        creator = kb.create_task(
            conn, title="review", assignee="red",
            workspace_kind="dir", workspace_path=str(shared),
        )
        own_dir = kb.create_task(
            conn, title="own dir", assignee="red", creator_task_id=creator,
            workspace_kind="dir", workspace_path=str(other),
        )
        opt_out = kb.create_task(
            conn, title="no shared checkout", assignee="red", creator_task_id=creator,
            workspace_kind="scratch",
        )
        own_row = kb.get_task(conn, own_dir)
        out_row = kb.get_task(conn, opt_out)
    assert own_row is not None and (own_row.workspace_kind, own_row.workspace_path) == ("dir", str(other))
    assert out_row is not None and (out_row.workspace_kind, out_row.workspace_path) == ("scratch", None)


def test_worktree_creator_does_not_pass_its_checkout_down(tmp_path, monkeypatch):
    """A worktree is per-task: a child must not inherit another task's checkout.

    Sharing one would put two tasks on the first task's branch (the same rule
    ``_insert_decomposed_child`` and ``_resolve_worktree_workspace`` enforce).
    """
    _isolated_home(tmp_path, monkeypatch)
    checkout = tmp_path / "repo" / ".worktrees" / "t_creator"
    checkout.mkdir(parents=True)
    with kbc.connect() as conn:
        creator = kb.create_task(
            conn, title="worktree creator", assignee="red",
            workspace_kind="worktree", workspace_path=str(checkout),
        )
        child = kb.create_task(conn, title="child", assignee="red", creator_task_id=creator)
        row = kb.get_task(conn, child)
    assert row is not None
    assert (row.workspace_kind, row.workspace_path) == ("scratch", None)


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


# ---------------------------------------------------------------------------
# Sibling boards: the gate must cover the whole kanban home, not one DB
# ---------------------------------------------------------------------------

def _two_named_boards() -> None:
    """Create boards ``alpha`` and ``beta`` under the isolated home."""
    from hermes_cli.kanban_db_boards import create_board

    for slug in ("alpha", "beta"):
        create_board(slug)


def test_live_claim_on_a_sibling_board_defers_the_same_workspace(tmp_path, monkeypatch):
    """A live claim on ANOTHER board's DB must serialize the same checkout.

    ``workspace_claim_holder`` used to scan only the dispatching board's
    ``conn``: board beta's tick could not see board alpha's live claim on the
    same ``dir`` checkout, so the two workers interleaved edits in one tree.
    """
    _isolated_home(tmp_path, monkeypatch)
    _two_named_boards()
    shared = tmp_path / "one-worktree"
    shared.mkdir()
    spawn = _Spawn()
    with kbc.connect(board="alpha") as conn_a:
        holder = kb.create_task(
            conn_a, title="holder", assignee="red",
            workspace_kind="dir", workspace_path=str(shared),
        )
        _hold_claim(conn_a, holder)
    with kbc.connect(board="beta") as conn_b:
        waiter = kb.create_task(
            conn_b, title="waiter", assignee="red",
            workspace_kind="dir", workspace_path=str(shared),
        )
        res = kbd.dispatch_once(
            conn_b, spawn_fn=spawn, reconcile_orphans=False, board="beta",
        )
        deferred = kb.get_task(conn_b, waiter)
        events = [(e.kind, e.payload) for e in kb.list_events(conn_b, waiter)]
        guard = kbd.check_respawn_guard(conn_b, waiter)
    assert waiter not in spawn.ids, f"spawned beside a sibling-board claim: {spawn.calls}"
    assert [tid for tid, _w, _p in res.skipped_workspace_busy] == [waiter]
    assert res.skipped_workspace_busy[0][2] == holder
    assert res.skipped_workspace_unknown == []
    assert deferred is not None and deferred.status == "ready"
    assert deferred.claim_lock is None and deferred.consecutive_failures == 0
    assert guard is None, guard
    busy = [p or {} for kind, p in events if kind == "workspace_busy"]
    assert busy, [kind for kind, _p in events]
    assert busy[0].get("holder_task_id") == holder
    assert busy[0].get("holder_board") == "alpha"
    assert busy[0].get("workspace") == str(shared)

    # Control: the SAME fixture spawns once the holder's claim is released, so
    # the deferral above cannot pass vacuously.
    with kbc.connect(board="alpha") as conn_a:
        _release(conn_a, holder)
    with kbc.connect(board="beta") as conn_b:
        res2 = kbd.dispatch_once(
            conn_b, spawn_fn=spawn, reconcile_orphans=False, board="beta",
        )
    assert [tid for tid, _w, _p in res2.spawned] == [waiter]
    assert spawn.ids == [waiter]


def test_unreadable_sibling_board_defers_closed_as_unknown(tmp_path, monkeypatch):
    """An unreadable sibling DB must defer, not spawn.

    Fail closed: a false defer leaves the card visible in ``ready``; a false
    spawn corrupts a checkout.  The reason is distinct (``workspace_busy_unknown``)
    so an operator can tell "held" from "cannot tell", the card keeps its lane,
    and no failure is counted.
    """
    home = _isolated_home(tmp_path, monkeypatch)
    _two_named_boards()
    shared = tmp_path / "one-worktree"
    shared.mkdir()
    spawn = _Spawn()
    with kbc.connect(board="alpha") as conn_a:
        holder = kb.create_task(
            conn_a, title="holder", assignee="red",
            workspace_kind="dir", workspace_path=str(shared),
        )
        _hold_claim(conn_a, holder)
    # The default board's DB is a sibling of beta and cannot be read.
    (home / "kanban.db").write_bytes(b"this is not a sqlite database")
    with kbc.connect(board="beta") as conn_b:
        waiter = kb.create_task(
            conn_b, title="waiter", assignee="red",
            workspace_kind="dir", workspace_path=str(shared),
        )
        res = kbd.dispatch_once(
            conn_b, spawn_fn=spawn, reconcile_orphans=False, board="beta",
        )
        deferred = kb.get_task(conn_b, waiter)
        events = [(e.kind, e.payload) for e in kb.list_events(conn_b, waiter)]
        guard = kbd.check_respawn_guard(conn_b, waiter)
    assert waiter not in spawn.ids, f"spawned while a board db was unreadable: {spawn.calls}"
    assert [tid for tid, _w in res.skipped_workspace_unknown] == [waiter]
    assert res.skipped_workspace_busy == []
    assert deferred is not None and deferred.status == "ready"
    assert deferred.claim_lock is None and deferred.consecutive_failures == 0
    assert deferred.last_failure_error and "cannot tell" in deferred.last_failure_error
    # The stamped reason must stay free of the filesystem path: the respawn
    # guard matches quota/auth words in it (a path like ``auth-service`` would
    # park a card that is only WAITING).
    assert str(shared) not in (deferred.last_failure_error or "")
    assert guard is None, guard
    unknown = [p or {} for kind, p in events if kind == "workspace_busy_unknown"]
    assert unknown, [kind for kind, _p in events]
    assert unknown[0].get("workspace") == str(shared)

    # Discriminating control: with the unreadable sibling gone and the claim
    # released, the same fixture spawns.
    (home / "kanban.db").unlink()
    with kbc.connect(board="alpha") as conn_a:
        _release(conn_a, holder)
    with kbc.connect(board="beta") as conn_b:
        res2 = kbd.dispatch_once(
            conn_b, spawn_fn=spawn, reconcile_orphans=False, board="beta",
        )
    assert [tid for tid, _w, _p in res2.spawned] == [waiter]


def test_sibling_board_enumeration_failure_defers_closed_as_unknown(tmp_path, monkeypatch):
    """When the sibling DB set cannot even be listed, defer with the same
    distinct unknown reason instead of assuming the workspace is free."""
    _isolated_home(tmp_path, monkeypatch)
    shared = tmp_path / "one-worktree"
    shared.mkdir()
    spawn = _Spawn()

    def _unlistable(_conn):
        raise OSError("boards root unreadable")

    with kbc.connect() as conn:
        waiter = kb.create_task(
            conn, title="waiter", assignee="red",
            workspace_kind="dir", workspace_path=str(shared),
        )
        monkeypatch.setattr(kbw, "_sibling_board_db_files", _unlistable)
        res = kbd.dispatch_once(conn, spawn_fn=spawn, reconcile_orphans=False)
        deferred = kb.get_task(conn, waiter)
        kinds = [e.kind for e in kb.list_events(conn, waiter)]
        guard = kbd.check_respawn_guard(conn, waiter)
    assert waiter not in spawn.ids
    assert [tid for tid, _w in res.skipped_workspace_unknown] == [waiter]
    assert "workspace_busy_unknown" in kinds, kinds
    assert deferred is not None and deferred.status == "ready"
    assert deferred.consecutive_failures == 0
    assert guard is None, guard
