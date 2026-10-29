"""Tests for decompose_triage_task — the DB-layer atomic fan-out
from the triage column. LLM-free by design.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import _insert_decomposed_child, decompose_triage_task
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(conn, title="rough idea", body=None, assignee=None, tenant=None):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        tenant=tenant,
        triage=True,
    )


def _git_repo(path: Path) -> Path:
    """A real git repo (children anchor their worktrees on it)."""
    path.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init", "-q", str(path)], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "--allow-empty", "-m", "init"],
        check=True, env=env,
    )
    return path


def _routed_dir_root(conn, checkout: Path, title="routed into triage") -> str:
    """A card carrying a ``dir`` checkout that reaches triage WITHOUT being born
    there — the block-loop breaker / ``kanban_transfer`` / manual-move route, so
    ``creator_workspace_inheritance``'s ``born_in_triage`` refusal never saw it.
    """
    tid = kb.create_task(
        conn, title=title, assignee="worker",
        workspace_kind="dir", workspace_path=str(checkout),
    )
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))
    task = kb.get_task(conn, tid)
    assert task is not None and task.status == "triage"
    assert (task.workspace_kind, task.workspace_path) == ("dir", str(checkout))
    return tid


def test_decompose_creates_children_and_promotes_root(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="ship a feature")
        assert kb.get_task(conn, tid).status == "triage"

    children = [
        {"title": "research", "body": "look at prior art", "assignee": "researcher", "parents": []},
        {"title": "build it", "body": "write code", "assignee": "engineer", "parents": [0]},
    ]
    with kbc.connect() as conn:
        child_ids = decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 2

    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, child_ids[0])
        c1 = kb.get_task(conn, child_ids[1])

    # Root flipped to todo with orchestrator assignee, gated by children.
    assert root.status == "todo"
    assert root.assignee == "orchestrator"
    # First child has no internal parents → ready on recompute_ready.
    assert c0.status == "ready"
    assert c0.assignee == "researcher"
    # Second child has parents=[0] → stays in todo until c0 completes.
    assert c1.status == "todo"
    assert c1.assignee == "engineer"


def test_decompose_records_audit_comment_and_event(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn)
        child_ids = decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "task A", "assignee": "researcher"}],
            author="alice",
        )
    assert child_ids is not None

    with kbc.connect() as conn:
        comments = kb.list_comments(conn, tid)
        events = kb.list_events(conn, tid)

    assert comments
    assert any(ev.kind == "decomposed" for ev in events)


def test_decomposed_children_of_a_routed_dir_card_get_isolated_workspaces(
    kanban_home, tmp_path,
):
    """FORCED TRIGGER A (non-repo checkout): no child may carry the root's ``dir``.

    A card created ``ready`` with a ``dir`` checkout and later moved to triage
    still holds that path (``born_in_triage`` is False), so the decomposer itself
    must refuse to propagate it. No git repo behind the path → ``scratch``, a
    private directory, never the shared checkout.
    """
    live = tmp_path / "live-checkout"
    live.mkdir()
    with kbc.connect() as conn:
        root = _routed_dir_root(conn, live)

    with kbc.connect() as conn:
        child_ids = decompose_triage_task(
            conn, root, root_assignee="orch",
            children=[
                {"title": "child A", "assignee": "worker"},
                {"title": "child B", "assignee": "worker", "workspace_kind": "scratch"},
            ],
            author="decomposer",
        )
    assert child_ids and len(child_ids) == 2

    with kbc.connect() as conn:
        children = [kb.get_task(conn, cid) for cid in child_ids]
    for child in children:
        assert child is not None
        assert child.workspace_kind != "dir", "a routed card's decomposition leaked a dir kind"
        assert child.workspace_path != str(live), "child landed in the shared live checkout"
        assert (child.workspace_kind, child.workspace_path) == ("scratch", None)


def test_decomposed_children_of_a_routed_dir_card_anchor_a_worktree(kanban_home, tmp_path):
    """FORCED TRIGGER A (repo-backed checkout): the isolated form is a worktree.

    Preferred over ``scratch`` because the children get the real codebase while
    staying off the live tree: dispatch materializes
    ``<repo>/.worktrees/<child-id>`` from the stored anchor.
    """
    live = _git_repo(tmp_path / "live-checkout")
    (live / "src.py").write_text("x = 1\n")
    with kbc.connect() as conn:
        root = _routed_dir_root(conn, live)

    with kbc.connect() as conn:
        child_ids = decompose_triage_task(
            conn, root, root_assignee="orch",
            children=[{"title": "child A", "assignee": "worker"}],
            author="decomposer",
        )
    assert child_ids

    with kbc.connect() as conn:
        child = kb.get_task(conn, child_ids[0])
    assert child is not None
    assert child.workspace_kind == "worktree"
    assert child.workspace_path == str(live.resolve())
    assert child.workspace_path != str(live / "src.py")
    assert child.branch_name is None, "siblings must not share the root's branch"


def test_anchored_worktree_child_dispatches_off_the_live_tree(kanban_home, tmp_path):
    """The stored shape must actually isolate at DISPATCH, not just look right.

    Dispatch resolves the anchor to ``<repo>/.worktrees/<child-id>`` — a fresh
    checkout per child, never the root's working tree (``materialize=False``:
    the path a real tick would use, without touching the disk).
    """
    from hermes_cli.kanban_db_workspace import _resolve_worktree_workspace

    live = _git_repo(tmp_path / "live-checkout")
    with kbc.connect() as conn:
        root = _routed_dir_root(conn, live)
        child_ids = decompose_triage_task(
            conn, root, root_assignee="orch",
            children=[
                {"title": "child A", "assignee": "worker"},
                {"title": "child B", "assignee": "worker"},
            ],
            author="decomposer",
        )
        assert child_ids and len(child_ids) == 2
        resolved = []
        for cid in child_ids:
            task = kb.get_task(conn, cid)
            assert task is not None and task.workspace_kind == "worktree"
            path, branch = _resolve_worktree_workspace(task, materialize=False)
            resolved.append((cid, task.workspace_path, str(path), branch))

    for cid, anchor, path, branch in resolved:
        assert anchor == str(live.resolve())
        assert path == str(live.resolve() / ".worktrees" / cid), path
        assert branch == f"wt/{cid}", "each sibling needs its own branch"
        assert path != str(live.resolve()), "child dispatch resolves onto the live tree"
    # Distinct checkouts per sibling: no shared writer, no shared branch.
    assert len({path for _, _, path, _ in resolved}) == 2
    assert len({branch for *_, branch in resolved}) == 2


def test_board_default_workdir_is_the_fallback_anchor(kanban_home, tmp_path):
    """No repo behind the root's path → the board's repo is the anchor.

    Dispatch's own anchor for a pathless worktree is the board
    ``default_workdir``; using it here keeps the child dispatchable instead of
    demoting a repo-backed fan-out to ``scratch``.
    """
    abandoned = tmp_path / "abandoned-checkout"
    abandoned.mkdir()
    board_repo = _git_repo(tmp_path / "board-repo")
    kb.write_board_metadata(kb.get_current_board(), default_workdir=str(board_repo))

    with kbc.connect() as conn:
        root = _routed_dir_root(conn, abandoned)
        child_ids = decompose_triage_task(
            conn, root, root_assignee="orch",
            children=[{"title": "child A", "assignee": "worker"}],
            author="decomposer",
        )
        assert child_ids
        child = kb.get_task(conn, child_ids[0])

    assert child is not None
    assert (child.workspace_kind, child.workspace_path) == ("worktree", str(board_repo.resolve()))


def test_explicit_dir_path_in_the_payload_still_wins(kanban_home, tmp_path):
    """The decomposer may hand a child a checkout it chose — explicit wins."""
    live = tmp_path / "live-checkout"
    live.mkdir()
    chosen = tmp_path / "chosen-checkout"
    chosen.mkdir()
    with kbc.connect() as conn:
        root = _routed_dir_root(conn, live)

    with kbc.connect() as conn:
        child_ids = decompose_triage_task(
            conn, root, root_assignee="orch",
            children=[{
                "title": "child A", "assignee": "worker",
                "workspace_kind": "dir", "workspace_path": str(chosen),
            }],
            author="decomposer",
        )
    assert child_ids
    with kbc.connect() as conn:
        child = kb.get_task(conn, child_ids[0])
    assert child is not None
    assert (child.workspace_kind, child.workspace_path) == ("dir", str(chosen))


def test_dir_inheritance_is_unchanged_for_a_root_outside_triage(kanban_home, tmp_path):
    """Control B: the deliberate shared-checkout case must keep working.

    ``decompose_triage_task`` only ever runs on a triage root, so this pins the
    sibling branch at the unit that owns the decision: with the root in ``todo``
    the child inherits the ``dir`` verbatim, exactly as before.
    """
    live = tmp_path / "shared-checkout"
    live.mkdir()
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="shared checkout root", assignee="worker",
            workspace_kind="dir", workspace_path=str(live),
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (tid,))
        root_row = conn.execute(
            "SELECT id, status, tenant, workspace_kind, workspace_path "
            "FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert root_row["status"] == "todo"
        with kb.write_txn(conn):
            child_id = _insert_decomposed_child(
                conn, tid, root_row,
                {"title": "child A", "assignee": "worker"}, "decomposer", int(time.time()),
            )

    with kbc.connect() as conn:
        child = kb.get_task(conn, child_id)
    assert child is not None
    assert (child.workspace_kind, child.workspace_path) == ("dir", str(live))
