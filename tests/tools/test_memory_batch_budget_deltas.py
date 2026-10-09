"""Budget-failure per-op delta hardening (#observed 2026-10-09: an agent burned its
consolidation-failure budget re-deriving which batch op caused the over-limit; the
failure carried totals but not per-op NET effects, so every retry re-guessed).

Contract: when apply_batch fails the budget check, the error names EACH op's net
char effect on the final state (adds minus removals), so one retry suffices.
"""

import pytest

from tools.memory_tool import MemoryStore


@pytest.fixture()
def store(monkeypatch, tmp_path):
    """Isolated store: fresh dir via get_memory_dir monkeypatch, fixed entries."""
    import tools.memory_tool as mt
    monkeypatch.setattr(mt, "get_memory_dir", lambda: tmp_path)
    s = MemoryStore(memory_char_limit=120, user_char_limit=1375)
    # _mutate re-reads disk each call — persist the fixture state through the
    # store's own write path so the store under test owns its data.
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "MEMORY.md").write_text("A" * 50 + "\n§\n" + "B" * 50)
    return s


def test_budget_failure_lists_per_op_deltas(store):
    # replace A(50) with X(90): net +40 → 103 + 40 = 143 > 120
    resp = store.apply_batch("memory", [
        {"action": "replace", "old_text": "A" * 50, "content": "X" * 90},
])
    body = str(resp)
    assert resp["success"] is False
    assert "net" in body.lower() or "delta" in body.lower(), body


def test_budget_failure_delta_signs_are_actionable(store):
    resp = store.apply_batch("memory", [
        {"action": "replace", "old_text": "A" * 50, "content": "X" * 140},  # +90
        {"action": "replace", "old_text": "B" * 50, "content": "Y" * 10},   # -40
    ])
    body = str(resp)
    # one op must be reported as ADDING (the offender) and one as removing
    assert "+" in body and "-" in body, body


def test_ok_batch_has_no_delta_noise(store):
    resp = store.apply_batch("memory", [
        {"action": "replace", "old_text": "A" * 50, "content": "A2" * 5},   # -40, fits
    ])
    assert resp["success"] is True
    assert "delta" not in str(resp).lower()
