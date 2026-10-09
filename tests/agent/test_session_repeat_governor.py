"""Session-scope Repeat Governor (R1) — forced-trigger tests.

These were written BEFORE the implementation and FAILED on it (forced trigger,
evidence-discipline): the per-turn controller wipes state in ``reset_for_turn()``
at every turn start, so cross-turn repeats (83% of red's 377 repeats) were
invisible. Each test below simulates the cross-turn sequence explicitly by
calling ``reset_for_turn()`` between turns and then asserts the session-scope
behavior the review mandates (review card t_800a8f96, changes 1-7).

Contracts under test:
- T1: identical read in turn N+1, identical result -> session-scope notice.
- T2: same call a third time -> block with a reference to the earlier result.
- T3: two identical failures with an embedded mutating counter ("failed 4
  times" -> "failed 5 times") -> same failure-class hash -> refusal cites the
  previous payload class.
- T4: identical failure, then a successful write_file -> counter resets.
- T5: poller (process_manage wait) unchanged result across turns -> no notice.
- T6: governor disabled via config -> zero decisions.
- T7: re-read after a successful patch to the same path -> allowed.
- T8: cross-turn repeat below session.warn_after -> allowed.
"""

import pytest

from agent.tool_guardrails import (
    SESSION_GOVERNOR_BUFFER_SIZE,
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
)


def _turns(controller, n, tool="read_file", args=None, result="same content", failed=False):
    """Observe the same call in n consecutive turns (reset_for_turn between each)."""
    decisions = []
    for _ in range(n):
        controller.reset_for_turn()
        decisions.append(controller.session_observe(tool, args or {"path": "/a"}, result, failed=failed))
    return decisions


def _turn(controller, tool, args, result, failed=False):
    """One in-turn observation (no reset) — for multi-call turn shapes."""
    return controller.session_observe(tool, args, result, failed=failed)


# ── T1: cross-turn identical read with identical result -> session notice ──


def test_t1_identical_read_next_turn_fires_session_notice():
    c = ToolCallGuardrailController()
    first = c.session_observe("read_file", {"path": "/a"}, "same content")
    assert first.action == "allow" and not first.message
    c.reset_for_turn()
    second = c.session_observe("read_file", {"path": "/a"}, "same content")
    assert second.action == "warn"
    assert "session" in second.message.lower()


def test_t1_within_turn_still_uses_per_turn_streak_only():
    """De-overlap gate (review change 1): repeats inside ONE turn get no
    session-scope decision — the existing per-turn stub/notice handles them."""
    c = ToolCallGuardrailController()
    d1 = _turn(c, "read_file", {"path": "/a"}, "same content")
    d2 = _turn(c, "read_file", {"path": "/a"}, "same content")
    assert d1.action == "allow" and d2.action == "allow"


# ── T2: third identical call -> block with reference to earlier result ──


def test_t2_third_identical_call_blocks_with_reference():
    c = ToolCallGuardrailController()
    _turns(c, 2)
    c.reset_for_turn()
    third = c.session_observe("read_file", {"path": "/a"}, "same content")
    assert third.action == "block"
    assert "turn" in third.message.lower()


# ── T3: mutating-counter failure payloads hash to the same failure class ──


def test_t3_mutating_error_counter_same_failure_class():
    c = ToolCallGuardrailController()
    payload_4 = '{"error":"Memory consolidation failed 4 times this turn. Stop retrying memory calls"}'
    payload_5 = '{"error":"Memory consolidation failed 5 times this turn. Stop retrying memory calls"}'
    first = _turn(c, "memory", {"action": "add", "content": "x"}, payload_4, failed=True)
    assert first.action == "allow"
    c.reset_for_turn()  # cross-turn boundary
    second = _turn(c, "memory", {"action": "add", "content": "x"}, payload_5, failed=True)
    assert second.action in {"warn", "block"}
    assert second.message  # carries the previous failure class, not a fresh start


def test_t3_failure_refusal_carries_guardrail_refusal_key():
    """Self-refusal hygiene (review change 5): the synthetic refusal must carry
    ``guardrail_refusal`` so the classifiers don't feed the streak it breaks."""
    from agent.tool_result_classification import is_guardrail_refusal
    from agent.tool_guardrails import toolguard_synthetic_result

    c = ToolCallGuardrailController()
    payload_4 = '{"error":"failed 4 times"}'
    payload_5 = '{"error":"failed 5 times"}'
    _turn(c, "memory", {"action": "add"}, payload_4, failed=True)
    c.reset_for_turn()  # cross-turn boundary
    second = _turn(c, "memory", {"action": "add"}, payload_5, failed=True)
    if second.should_halt:
        assert is_guardrail_refusal(toolguard_synthetic_result(second))


# ── T4: successful mutation resets the failure counter ──


def test_t4_successful_write_resets_failure_counter():
    c = ToolCallGuardrailController()
    payload_4 = '{"error":"failed 4 times"}'
    payload_5 = '{"error":"failed 5 times"}'
    _turn(c, "memory", {"action": "add"}, payload_4, failed=True)
    c.reset_for_turn()  # cross-turn boundary
    _turn(c, "memory", {"action": "add"}, payload_5, failed=True)
    c.reset_for_turn()
    # A successful write_file is progress; the next identical failure is a new experiment.
    _turn(c, "write_file", {"path": "/b", "content": "x"}, '{"bytes_written": 1}', failed=False)
    c.reset_for_turn()
    third = _turn(c, "memory", {"action": "add"}, '{"error":"failed 6 times"}', failed=True)
    assert third.action == "allow"


# ── T5: pollers exempt from the session SUCCESS-repeat tier only ──


def test_t5_poller_identical_success_no_session_notice():
    c = ToolCallGuardrailController()
    decisions = _turns(c, 4, tool="process_manage", args={"action": "wait", "session_id": "p1"},
                       result='{"status":"timeout","process_running":true}')
    assert all(d.action == "allow" for d in decisions)


def test_t5_poller_identical_failures_still_count():
    """A poller failing with the identical error class across turns must still
    count (12 wait timeouts -> re-poll is the audit's exact pattern)."""
    c = ToolCallGuardrailController()
    err = '{"error":"process not found"}'
    first = _turn(c, "process_manage", {"action": "wait", "session_id": "p1"}, err, failed=True)
    assert first.action == "allow"
    c.reset_for_turn()  # cross-turn boundary
    second = _turn(c, "process_manage", {"action": "wait", "session_id": "p1"}, err, failed=True)
    assert second.action in {"warn", "block"}


# ── T6: master switch + 0 = disabled ──


def test_t6_disabled_governor_makes_no_decisions():
    c = ToolCallGuardrailController(
        ToolCallGuardrailConfig(session_governor_enabled=False)
    )
    decisions = _turns(c, 5)
    assert all(d.action == "allow" for d in decisions)


def test_t6_zero_threshold_disables_tier():
    c = ToolCallGuardrailController(
        ToolCallGuardrailConfig(session_warn_after=0, session_block_after=0)
    )
    decisions = _turns(c, 5)
    assert all(d.action == "allow" for d in decisions)


# ── T7: re-read after successful patch to the same path is allowed ──


def test_t7_progress_reset_allows_legitimate_reread():
    c = ToolCallGuardrailController()
    _turns(c, 2)
    c.reset_for_turn()
    _turn(c, "patch", {"path": "/a", "patch": "x"}, '{"success": true}', failed=False)
    c.reset_for_turn()
    reread = c.session_observe("read_file", {"path": "/a"}, "same content")
    assert reread.action == "allow"


def test_t7_progress_reset_clears_failure_counter_too():
    c = ToolCallGuardrailController()
    _turn(c, "read_file", {"path": "/a"}, "x", failed=True)
    c.reset_for_turn()
    _turn(c, "patch", {"path": "/a", "patch": "x"}, '{"success": true}', failed=False)
    c.reset_for_turn()
    retry = c.session_observe("read_file", {"path": "/a"}, "y", failed=False)
    assert retry.action == "allow"


# ── T8: repeat below session.warn_after is allowed ──


def test_t8_below_warn_threshold_allowed():
    c = ToolCallGuardrailController(
        ToolCallGuardrailConfig(session_warn_after=5, session_block_after=0)
    )
    decisions = _turns(c, 4)
    assert all(d.action == "allow" for d in decisions)


# ── config plumbing ──


def test_session_governor_config_from_mapping():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "session": {"warn_after": 4, "block_after": 7},
            "session_governor_enabled": False,
        }
    )
    assert cfg.session_warn_after == 4
    assert cfg.session_block_after == 7
    assert cfg.session_governor_enabled is False


def test_session_governor_config_zero_disables():
    cfg = ToolCallGuardrailConfig.from_mapping({"session": {"warn_after": 0, "block_after": 0}})
    assert cfg.session_warn_after == 0
    assert cfg.session_block_after == 0


def test_session_governor_defaults_off_thresholds_on():
    cfg = ToolCallGuardrailConfig.from_mapping({})
    assert cfg.session_governor_enabled is True
    assert cfg.session_warn_after > 0
    assert cfg.session_block_after > cfg.session_warn_after


# ── A1: persisted-path pointers are session-scoped with the governor ──


def test_a1_persisted_paths_survive_reset_for_turn():
    c = ToolCallGuardrailController()
    c.record_persisted_result("call-1", "/tmp/spill-1.md")
    c.reset_for_turn()
    assert c._persisted_result_paths.get("call-1") == "/tmp/spill-1.md"


def test_a1_persisted_paths_cleared_by_reset_for_session():
    c = ToolCallGuardrailController()
    c.record_persisted_result("call-1", "/tmp/spill-1.md")
    c.reset_for_session()
    assert not c._persisted_result_paths


# ── A2: multimodal / non-string results hash instead of being exempt ──


def test_a2_non_string_results_form_session_streaks():
    c = ToolCallGuardrailController()
    result = {"content": [{"type": "image", "url": "x"}], "text_summary": "same"}
    d1 = _turn(c, "browser_snapshot", {}, result)
    c.reset_for_turn()
    d2 = _turn(c, "browser_snapshot", {}, {"content": [{"type": "image", "url": "x"}], "text_summary": "same"})
    assert d1.action == "allow"
    assert d2.action == "warn"


# ── A3: count-before-gate ordering — the first blocked call is recorded ──


def test_a3_blocked_call_is_still_recorded():
    """The session buffer must record the call BEFORE the block decision, so a
    blocked repeat still advances the count (same ordering as _check_loop_cap)."""
    c = ToolCallGuardrailController()
    _turns(c, 3)  # 1st ok, 2nd notice, 3rd block
    c.reset_for_turn()
    fourth = c.session_observe("read_file", {"path": "/a"}, "same content")
    assert fourth.action == "block"
    assert fourth.count == 4


# ── ring buffer stays bounded ──


def test_ring_buffer_bounded():
    c = ToolCallGuardrailController()
    for i in range(SESSION_GOVERNOR_BUFFER_SIZE + 50):
        c.session_observe("web_search", {"query": f"q{i}"}, f"result {i}")
    assert len(c._session_buffer) == SESSION_GOVERNOR_BUFFER_SIZE


# ── session reset lives outside reset_for_turn ──


def test_reset_for_session_clears_governor_state():
    c = ToolCallGuardrailController()
    _turns(c, 3)
    c.reset_for_session()
    d = c.session_observe("read_file", {"path": "/a"}, "same content")
    assert d.action == "allow"
