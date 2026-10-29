"""A card born in ``triage`` must not inherit its creator's ``dir`` checkout.

History (red7, 2026-10-29): a reviewer worker running in ``dir /home/island/ideas/red7``
(the LIVE checkout) filed a follow-up card with no workspace. The card inherited that
path, was born in ``triage``, and ``auto_decompose`` split it six seconds later into
three children that all inherited the same live checkout — automated workers editing
production code during an operator acceptance run.

A triage card is a PROPOSAL: the decomposer decides its children's workspaces anyway, so
pinning a live checkout to it is the defect. Worktree inheritance stays allowed (a
worktree is already isolated); only ``dir`` inheritance is refused for triage-born cards.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _live_creator(conn, live_path: str) -> str:
    """A claimed task whose workspace is a shared ``dir`` checkout."""
    tid = kb.create_task(
        conn, title="reviewer in live checkout", assignee="worker",
        workspace_kind="dir", workspace_path=live_path,
    )
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid


def test_triage_child_does_not_inherit_the_live_dir(kanban_home: Path, tmp_path: Path) -> None:
    live = str(tmp_path / "live-checkout")
    with kbc.connect_closing() as conn:
        creator = _live_creator(conn, live)

        child = kb.create_task(
            conn, title="follow-up", assignee="worker",
            creator_task_id=creator, triage=True,
        )

        task = kb.get_task(conn, child)
        assert task is not None
        assert task.workspace_kind != "dir", "triage card inherited a live checkout"
        assert task.workspace_path != live


def test_non_triage_child_still_inherits_the_creator_checkout(kanban_home: Path, tmp_path: Path) -> None:
    """Control: the 2026-10-27 collision fix must keep working for normal cards."""
    live = str(tmp_path / "live-checkout")
    with kbc.connect_closing() as conn:
        creator = _live_creator(conn, live)

        child = kb.create_task(
            conn, title="follow-up", assignee="worker",
            creator_task_id=creator, triage=False,
        )

        task = kb.get_task(conn, child)
        assert task is not None
        assert (task.workspace_kind, task.workspace_path) == ("dir", live)
