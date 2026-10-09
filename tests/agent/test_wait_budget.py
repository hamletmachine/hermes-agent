"""R2 — agent-layer in-turn wait budget (forced-trigger tests).

Review mandate: enforce the cumulative wait budget at the AGENT layer
(guardrails/dispatch wrapper), NOT inside ``process_registry.wait()`` — wait()
has no turn context. Two tiers: WARN at ~100s cumulative, REFUSE at ~150s. The
refuse result cites the notify/wake contract (process_registry timeout_note
text: "you will be notified on exit — do more work instead of waiting again").
Poller wait failures still count (not exempt).

These tests FAILED on the pre-change code: no wait tracking existed.
"""

import pytest

from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    TOOL_WAIT_BUDGET_WARN_SECONDS,
    TOOL_WAIT_BUDGET_REFUSE_SECONDS,
)


def _record_wait(controller, seconds, failed=False):
    """The runtime seam: a process_manage wait result reports its window."""
    return controller.observe_wait_window(seconds, failed=failed)


def test_wait_budget_tiers_exist_at_reviewed_magnitudes():
    assert TOOL_WAIT_BUDGET_WARN_SECONDS == 100
    assert TOOL_WAIT_BUDGET_REFUSE_SECONDS == 150


def test_first_long_wait_below_warn_is_allowed():
    c = ToolCallGuardrailController()
    d = _record_wait(c, 90)
    assert d is None


def test_warn_fires_at_100s_cumulative():
    c = ToolCallGuardrailController()
    assert _record_wait(c, 60) is None
    d = _record_wait(c, 60)  # cumulative 120 >= 100
    assert d is not None and d.action == "warn"
    assert "notify" in d.message.lower() or "end the turn" in d.message.lower()


def test_refuse_fires_at_150s_cumulative_and_cites_wake_contract():
    c = ToolCallGuardrailController()
    _record_wait(c, 80)
    d2 = _record_wait(c, 60)  # 140: warn tier
    assert d2.action == "warn"
    d = _record_wait(c, 60)  # 200 >= 150
    assert d.action == "block"
    assert "notify" in d.message.lower() and "do not wait" in d.message.lower()


def test_wait_budget_resets_per_turn():
    c = ToolCallGuardrailController()
    _record_wait(c, 90)
    _record_wait(c, 90)  # warn fired
    c.reset_for_turn()
    assert _record_wait(c, 90) is None  # fresh turn: budget reset


def test_poller_wait_failures_still_count():
    c = ToolCallGuardrailController()
    assert _record_wait(c, 80, failed=True) is None
    d = _record_wait(c, 80, failed=True)  # 160 >= 150
    assert d.action == "block"


def test_short_waits_accumulate_too():
    c = ToolCallGuardrailController()
    for _ in range(2):
        assert _record_wait(c, 45) is None  # 90 cumulative — below warn
    d2 = _record_wait(c, 45)  # 135: warn
    assert d2.action == "warn"
    d = _record_wait(c, 45)  # 180 >= 150
    assert d.action == "block"