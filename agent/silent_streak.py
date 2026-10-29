"""R4 — silent-streak checkpoint.

92.8% of red's iterations were silent (no user-visible text), longest chain 331.
A static N=40 is wrong fleet-wide (median useful silent burst ~18, legitimate
work runs 100+), so the checkpoint scales with the turn's own iteration budget:
``N = clamp(0.15 x max_iterations, 20, 60)`` with ``agent.silent_streak_checkpoint``
as an explicit override that beats the computed value.

The NOTICE is one-shot and built on the cache-safe tail-append pattern (append to
the newest, not-yet-persisted tool result; per-turn flag) — never a synthetic
user/system row mid-turn (turn_iteration_prep contract).

The HARD END (cron/kanban platforms only) is a turn FINALIZE with the distinct
exit reason ``silent_streak_finalize`` — parallel to the persistence-failure
break path, NOT ``_toolguard_controlled_halt_response`` ("I stopped retrying"
would misreport a checkpoint as a failure).

Silence signal: a round whose ``turn_content`` is empty (the same housekeeping
classification turn_tool_round.stage_tool_call_message uses); any round with
user-visible text resets the streak.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SILENT_STREAK_FINALIZE_EXIT_REASON = "silent_streak_finalize"

_N_RATIO = 0.15
_N_MIN = 20
_N_MAX = 60

# Unattended platforms where a silent run must report progress and end.
#
# CRON ONLY — kanban was removed 2026-10-29. A kanban worker's contract is a
# TERMINAL TOOL CALL (kanban_complete / kanban_request_review / kanban_block).
# Ending its turn for silence cannot produce that call; it guarantees a run that
# exits rc=0 without one, which the dispatcher books as a protocol violation and
# retries - i.e. the checkpoint MANUFACTURED the crash class it was meant to
# prevent. Measured on this box: 13 worker runs reached the finalize and 11 of
# those cards carry crashed runs (t_f0c93a99 x6, t_736d0e16 x4, t_74e17583 x2).
# A silent kanban worker is already bounded by max_iterations and the task's
# max_runtime_seconds, and the one-shot NOTICE still tells it to checkpoint via
# kanban_comment. Attended surfaces keep the notice-only tier.
_HARD_END_PLATFORMS = frozenset({"cron"})

SILENT_STREAK_NOTICE_TEMPLATE = (
    "[SYSTEM NOTICE — silent-streak checkpoint] You have run {count} iterations without any "
    "user-visible text. Checkpoint durable progress now (kanban_comment / memory / a status "
    "line), then continue or wrap up: an unattended run that stays silent cannot be audited "
    "and may be ended at the next checkpoint."
)

# Agent attributes the tracker owns (also reset per turn where noted).
_SILENT_STREAK_STATE: tuple[tuple[str, Any], ...] = (
    ("_silent_round_count", 0),
    ("_silent_checkpoint_notice_shown", False),
)


def silent_streak_n(max_iterations: int, override: Any = None) -> int:
    """Checkpoint interval: clamp(0.15 x max_iterations, 20, 60); an explicit
    ``agent.silent_streak_checkpoint`` override is absolute when positive."""
    if isinstance(override, (int, float)) and not isinstance(override, bool) and override > 0:
        return int(override)
    return max(_N_MIN, min(_N_MAX, int(_N_RATIO * max(1, max_iterations))))


def compute_silent_streak_checkpoint(agent: Any) -> int:
    """N for this agent: the explicit override beats the computed value."""
    return silent_streak_n(getattr(agent, "max_iterations", 0) or 0, getattr(agent, "silent_streak_checkpoint", None))


def _ensure_state(agent: Any) -> None:
    for name, default in _SILENT_STREAK_STATE:
        if not hasattr(agent, name):
            setattr(agent, name, default)


def note_round_visibility(agent: Any, *, silent: bool) -> None:
    """Track one assistant round: silent = no user-visible text (turn_content
    empty). Any visible round resets the streak."""
    _ensure_state(agent)
    if not silent:
        agent._silent_round_count = 0
        return
    agent._silent_round_count += 1
    n = compute_silent_streak_checkpoint(agent)
    if agent._silent_round_count >= n:
        agent._silent_checkpoint_notice_shown = True


def should_finalize_silent_streak(agent: Any) -> bool:
    """Whether the streak reached the checkpoint this turn (notice tier)."""
    _ensure_state(agent)
    n = compute_silent_streak_checkpoint(agent)
    return agent._silent_round_count >= n


def silent_streak_checkpoint_due(agent: Any) -> bool:
    """One-shot notice due NOW: streak reached N and the notice has not been
    shown this turn."""
    _ensure_state(agent)
    if agent._silent_checkpoint_notice_shown:
        return False
    n = compute_silent_streak_checkpoint(agent)
    if agent._silent_round_count < n:
        return False
    agent._silent_checkpoint_notice_shown = True
    return True


def silent_streak_hard_end_eligible(agent: Any) -> bool:
    """Cron only: the streak reached the checkpoint on an unattended surface that
    has NO terminal-call contract. Kanban workers are excluded — see
    ``_HARD_END_PLATFORMS`` for the measured reason."""
    _ensure_state(agent)
    if not should_finalize_silent_streak(agent):
        return False
    platform = str(getattr(agent, "platform", None) or "").strip().lower()
    return platform in _HARD_END_PLATFORMS


def reset_silent_streak_for_turn(agent: Any) -> None:
    """Per-turn reset of the one-shot notice flag (the streak count itself is
    intentionally NOT reset per turn — it spans rounds within the turn's
    iteration budget; visible text resets it)."""
    _ensure_state(agent)
    agent._silent_checkpoint_notice_shown = False


def reset_silent_streak_full(agent: Any) -> None:
    """Full reset (new turn with visible text / new session)."""
    for name, default in _SILENT_STREAK_STATE:
        setattr(agent, name, default)