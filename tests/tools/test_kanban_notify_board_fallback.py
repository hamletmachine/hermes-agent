"""Cards created without a gateway session still notify the board's operator.

Worker- and CLI-created cards have no ``HERMES_SESSION_PLATFORM`` /
``HERMES_SESSION_CHAT_ID`` ContextVars, so ``_resolve_notify_target`` returns
None and the card never gets a ``kanban_notify_subs`` row.  The board may name
its route-owning destination in ``board.json``; that fallback is what keeps
the operator's DM informed about cards the dispatcher creates.
"""
from __future__ import annotations

import json
from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_boards import board_metadata_path, create_board
from hermes_cli import kanban_db_notify as kbn
import tools.kanban_tools as kt


def _isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    return home


def _board_with_notify(slug: str = "red7") -> None:
    create_board(slug)
    path = board_metadata_path(slug)
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["notify"] = {
        "platform": "matrix",
        "chat_id": "!operator:matrix.org",
        "notifier_profile": "red",
        "chat_type": "dm",
        "delivery_mode": "notify+wake",
        "delivery_metadata": {"chat_type": "dm", "scope_id": "matrix.org"},
    }
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def test_worker_created_card_inherits_the_board_notify_target(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    _board_with_notify()
    monkeypatch.setattr(kt, "_resolve_notify_target", lambda: None)
    with kbc.connect(board="red7") as conn:
        tid = kb.create_task(conn, title="operator-visible card", assignee="red")
        ok = kt._maybe_auto_subscribe(conn, tid, board="red7")
        subs = kbn.list_notify_subs(conn, task_id=tid)
    assert ok is True
    assert len(subs) == 1, subs
    assert subs[0]["platform"] == "matrix"
    assert subs[0]["chat_id"] == "!operator:matrix.org"
    assert subs[0]["notifier_profile"] == "red"
    assert subs[0]["delivery_mode"] == "notify+wake"


def test_cli_helper_subscribes_the_board_target(tmp_path, monkeypatch):
    """The CLI create path calls the shared helper, not the tool handler."""
    _isolated_home(tmp_path, monkeypatch)
    _board_with_notify()
    with kbc.connect(board="red7") as conn:
        tid = kb.create_task(conn, title="cli card", assignee="red")
        ok = kbn.auto_subscribe_board(conn, tid, board="red7")
        subs = kbn.list_notify_subs(conn, task_id=tid)
    assert ok is True
    assert len(subs) == 1, subs
    assert subs[0]["chat_id"] == "!operator:matrix.org"


def test_no_board_target_keeps_the_old_no_subscription_behaviour(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    create_board("red7")
    monkeypatch.setattr(kt, "_resolve_notify_target", lambda: None)
    with kbc.connect(board="red7") as conn:
        tid = kb.create_task(conn, title="silent card", assignee="red")
        ok = kt._maybe_auto_subscribe(conn, tid, board="red7")
        subs = kbn.list_notify_subs(conn, task_id=tid)
    assert ok is False
    assert subs == []
