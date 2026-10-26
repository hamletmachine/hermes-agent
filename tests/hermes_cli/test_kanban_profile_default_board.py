"""Per-profile ``kanban.default_board`` resolution.

The kanban home is shared across profiles by design, so the on-disk ``current``
file cannot express "Red defaults to red7, Exsul to default". These tests pin the
resolution order: context override > HERMES_KANBAN_BOARD > profile config
default > shared current file > default.
"""
from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    return home


def _config(monkeypatch, default_board: str) -> None:
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"kanban": {"default_board": default_board}},
    )


def test_profile_default_board_is_used(fresh_home, monkeypatch):
    kb.create_board("proj")
    _config(monkeypatch, "proj")
    assert kb.get_current_board() == "proj"


def test_env_board_wins_over_profile_default(fresh_home, monkeypatch):
    kb.create_board("proj")
    kb.create_board("envboard")
    _config(monkeypatch, "proj")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "envboard")
    assert kb.get_current_board() == "envboard"


def test_context_override_wins_over_profile_default(fresh_home, monkeypatch):
    kb.create_board("proj")
    kb.create_board("ctxboard")
    _config(monkeypatch, "proj")
    with kb.scoped_current_board("ctxboard"):
        assert kb.get_current_board() == "ctxboard"


def test_invalid_profile_default_falls_through_to_current(fresh_home, monkeypatch):
    kb.create_board("onfile")
    kb.set_current_board("onfile")
    _config(monkeypatch, "does-not-exist")
    assert kb.get_current_board() == "onfile"


def test_empty_profile_default_keeps_existing_behavior(fresh_home, monkeypatch):
    kb.create_board("onfile")
    kb.set_current_board("onfile")
    _config(monkeypatch, "")
    assert kb.get_current_board() == "onfile"
