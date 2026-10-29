"""`boards switch` / `boards create --switch` must persist where resolution reads.

The per-profile-default landing (df8809c3b7) makes ``get_current_board()`` consult
the ACTIVE PROFILE's ``kanban.default_board`` before the shared
``<root>/kanban/current`` file. Every fleet profile sets that key, so a
shared-file-only write prints success and changes nothing: the documented
``boards switch`` contract is silently broken (see review t_11af5b7e, BLOCK).

DISCIPLINE: real on-disk profile configs throughout — this suite must never
stub ``load_config_readonly``, which is the blind spot the shipped resolution
tests carry (a wrong-profile read passes them).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import hermes_cli
from hermes_cli import kanban_boards as kbd
from hermes_cli import kanban_db as kb

WORKTREE = Path(__file__).resolve().parents[2]
assert str(Path(hermes_cli.__file__).resolve()) == str((WORKTREE / "hermes_cli" / "__init__.py").resolve()), (
    f"hermes_cli imported from {hermes_cli.__file__} — a worktree run silently validating the live tree"
)


@pytest.fixture
def shared_root(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    import hermes_constants

    hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    kb._INITIALIZED_PATHS.clear()
    return root


ALPHA_CONFIG = "kanban:\n  default_board: alpha  # workspace pin, do not drop\n"


def _profile(root, name: str, default_board: str) -> Path:
    home = root / "profiles" / name
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(f"kanban:\n  default_board: {default_board}\n", encoding="utf-8")
    return home


@pytest.fixture
def alpha(shared_root):
    home = shared_root / "profiles" / "alpha"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(ALPHA_CONFIG, encoding="utf-8")
    return home


@pytest.fixture
def become_home(request):
    """Enter the profile scope the production gateway uses: a context-local home
    override. kanban_home() stays on the shared process root (by design) while
    get_config_path() follows the override — exactly what the fix must honor."""
    import hermes_constants

    stack: list = []
    request.addfinalizer(
        lambda: [hermes_constants.reset_hermes_home_override(tok) for tok in reversed(stack)]
    )

    def become(home: Path) -> None:
        stack.append(hermes_constants.set_hermes_home_override(home))

    return become


def _become(become, home: Path) -> None:
    become(home)


def test_switch_persists_where_resolution_reads(shared_root, alpha, become_home):
    kb.create_board("beta")
    _become(become_home, alpha)

    rc = kbd._cmd_boards_switch(argparse.Namespace(slug="beta"))
    assert rc == 0
    assert kb.get_current_board() == "beta"
    written = (alpha / "config.yaml").read_text(encoding="utf-8")
    assert "default_board: beta" in written
    assert "# workspace pin" in written  # ruamel comment preservation through the write


def test_switch_does_not_touch_sibling_profile(shared_root, alpha, become_home):
    bravo = _profile(shared_root, "bravo", "bravo")
    bravo_before = (bravo / "config.yaml").read_text(encoding="utf-8")
    kb.create_board("beta")
    _become(become_home, alpha)
    assert kbd._cmd_boards_switch(argparse.Namespace(slug="beta")) == 0
    assert (bravo / "config.yaml").read_text(encoding="utf-8") == bravo_before


def test_create_switch_persists(shared_root, alpha, become_home):
    _become(become_home, alpha)
    rc = kbd._cmd_boards_create(
        argparse.Namespace(
            slug="gamma", switch=True, name="", description="", icon="", color="", default_workdir="",
        )
    )
    assert rc == 0
    assert kb.get_current_board() == "gamma"  # FAILS on HEAD
    assert "default_board: gamma" in (alpha / "config.yaml").read_text(encoding="utf-8")


def test_switch_refuses_archived_board(shared_root, alpha, become_home):
    # The archived guard must keep erroring: a success message over an ignored write is the defect.
    kb.create_board("beta")
    meta_path = kb.board_dir("beta") / "board.json"
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["archived"] = True
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    _become(become_home, alpha)
    before = (alpha / "config.yaml").read_text(encoding="utf-8")
    rc = kbd._cmd_boards_switch(argparse.Namespace(slug="beta"))
    assert rc != 0
    assert (alpha / "config.yaml").read_text(encoding="utf-8") == before
