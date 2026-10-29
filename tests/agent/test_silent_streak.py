"""R4 — silent-streak checkpoint (forced-trigger tests).

Review mandate:
- One-shot notice appended to the newest tool result (the cache-safe tail-append
  pattern from turn_iteration_prep._maybe_inject_iteration_budget_warning),
  flag reset per turn.
- N = clamp(0.15 x max_iterations, 20, 60); agent.silent_streak_checkpoint
  override beats the computed value.
- Hard-end for cron/kanban = turn FINALIZE with distinct exit reason
  ``silent_streak_finalize`` — NOT _toolguard_controlled_halt_response.
- Silence signal = turn_content empty per turn_tool_round's housekeeping
  classification; reset on any turn with user-visible text.

These FAILED on the pre-change code: nothing tracked silent streaks.
"""

import pytest

from agent.silent_streak import (
    SILENT_STREAK_FINALIZE_EXIT_REASON,
    compute_silent_streak_checkpoint,
    note_round_visibility,
    silent_streak_n,
    should_finalize_silent_streak,
)


# ── N computation ──────────────────────────────────────────────────────────


def test_n_clamped_to_20_for_small_budgets():
    assert silent_streak_n(100) == 20  # 0.15*100=15 -> clamp to 20


def test_n_scales_and_caps_at_60():
    assert silent_streak_n(500) == 60  # 0.15*500=75 -> clamp to 60
    assert silent_streak_n(200) == 30  # in-range: exact 0.15x


def test_explicit_override_beats_computed():
    assert silent_streak_n(500, override=7) == 7
    assert silent_streak_n(100, override=200) == 200  # override is absolute


def test_compute_from_agent():
    agent = type("A", (), {"max_iterations": 500})()
    assert compute_silent_streak_checkpoint(agent) == 60
    agent2 = type("A", (), {"max_iterations": 500, "silent_streak_checkpoint": 11})()
    assert compute_silent_streak_checkpoint(agent2) == 11


# ── streak tracking ────────────────────────────────────────────────────────


def test_silent_rounds_accumulate_and_visible_resets():
    agent = type("A", (), {"max_iterations": 500, "silent_streak_checkpoint": None})()
    for _ in range(59):
        note_round_visibility(agent, silent=True)
    assert not should_finalize_silent_streak(agent)
    note_round_visibility(agent, silent=True)  # 60
    assert should_finalize_silent_streak(agent)
    note_round_visibility(agent, silent=False)  # user-visible text resets
    assert not should_finalize_silent_streak(agent)
    assert agent._silent_round_count == 0


def test_notice_fires_once_per_streak():
    agent = type("A", (), {"max_iterations": 100, "silent_streak_checkpoint": None})()
    for _ in range(19):
        note_round_visibility(agent, silent=True)
    assert agent._silent_checkpoint_notice_shown is False
    note_round_visibility(agent, silent=True)  # 20 = N: notice fires
    assert agent._silent_checkpoint_notice_shown is True
    # One-shot: further silent rounds do not re-fire.
    note_round_visibility(agent, silent=True)
    assert agent._silent_checkpoint_notice_shown is True


def test_flag_resets_per_turn():
    agent = type("A", (), {"max_iterations": 100})()
    note_round_visibility(agent, silent=True)
    agent._silent_checkpoint_notice_shown = True  # what the one-shot does
    from agent.silent_streak import reset_silent_streak_for_turn
    reset_silent_streak_for_turn(agent)
    assert agent._silent_checkpoint_notice_shown is False


# ── hard-end finalization (cron/kanban only) ───────────────────────────────


def test_finalize_gated_to_unattended_platforms():
    from agent.silent_streak import silent_streak_hard_end_eligible
    for platform in ("cron",):
        agent = type("A", (), {"platform": platform, "max_iterations": 100})()
        for _ in range(60):
            note_round_visibility(agent, silent=True)
        assert silent_streak_hard_end_eligible(agent)
    # kanban joined this list 2026-10-29: a worker's contract is a TERMINAL call, so
    # ending its turn for silence manufactured protocol violations (13 runs, 11 cards
    # with crashed retries). The one-shot notice still applies there.
    for platform in ("cli", "tui", "matrix", "telegram", "kanban"):
        agent = type("A", (), {"platform": platform, "max_iterations": 100})()
        for _ in range(60):
            note_round_visibility(agent, silent=True)
        assert not silent_streak_hard_end_eligible(agent)


def test_finalize_exit_reason_is_distinct():
    assert SILENT_STREAK_FINALIZE_EXIT_REASON == "silent_streak_finalize"
    from agent.turn_failure_copy import exit_reason_failure
    # Parallel to the persistence-failure break path: advisory, not a failure verdict
    # (the work didn't fail; it ran silent and must report).
    verdict = exit_reason_failure(SILENT_STREAK_FINALIZE_EXIT_REASON)
    assert verdict is None or not getattr(verdict, "fails_turn", True)
