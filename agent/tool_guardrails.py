"""Pure tool-call loop guardrail primitives.

The controller is side-effect free: it tracks per-turn tool-call observations
and returns decisions. Runtime code decides whether a decision becomes warning
guidance, a synthetic tool result, or a controlled turn halt.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

from utils import safe_json_loads
from agent.tool_result_classification import (
    GUARDRAIL_REFUSAL_KEY,
    file_mutation_result_landed,
    is_guardrail_refusal,
)


IDEMPOTENT_TOOL_NAMES = frozenset({
    "read_file", "search_files", "web_search", "web_extract", "session_search", "skill_view", "skills_list",
    "browser_snapshot", "browser_console", "browser_get_images", "mcp_filesystem_read_file",
    "mcp_filesystem_read_text_file", "mcp_filesystem_read_multiple_files", "mcp_filesystem_list_directory",
    "mcp_filesystem_list_directory_with_sizes", "mcp_filesystem_directory_tree", "mcp_filesystem_get_file_info",
    "mcp_filesystem_search_files",
})

MUTATING_TOOL_NAMES = frozenset({
    "terminal", "execute_code", "write_file", "patch", "todo_list", "memory", "skill_manage",
    "browser_click", "browser_type", "browser_press", "browser_scroll", "browser_navigate",
    "send_message", "cronjob_manage", "delegate_task", "process_manage",
})

# Pollers: legitimately re-invoked with identical args; the identical-call NOTICE never fires.
STALL_GUARD_REPEATABLE_TOOLS = frozenset({"process_manage"})
_STALL_GUARD_REPEATABLE_SUFFIXES = ("_get_result", "_poll")  # generated / MCP poller conventions
# Nth consecutive identical (tool, args, result) call that fires the notice; 3 tolerates one double-check.
STALL_GUARD_IDENTICAL_CALL_THRESHOLD = 3
# Repeating multi-call cycles (A,B,A,B,... with identical args AND results) defeat the
# consecutive streak above — every alternation resets it, so a model replaying the same
# 2–4 call batch each iteration ran to the budget unflagged (port of can1357/oh-my-pi#10521,
# which widened their loop guard from single-call turns to whole tool-call batches).
# Longest cycle period detected; laps reuse the streak thresholds (notice at
# STALL_GUARD_IDENTICAL_CALL_THRESHOLD laps, halt at no_progress_block_after laps).
_STALL_GUARD_MAX_CYCLE_PERIOD = 4
# History window: enough for block_after laps of the longest cycle plus slack.
_STALL_GUARD_CYCLE_HISTORY = 64
# From the 2nd byte-identical repeat the duplicate payload becomes a reference stub; smaller results
# aren't worth it, errors never are. The args preview keeps WHAT was called if compression evicts the original.
IDENTICAL_RESULT_STUB_MIN_CHARS = 512
_RESULT_STUB_ARGS_PREVIEW_CHARS = 120

# Tools whose "failure" is normal work output (red test run, empty grep, page timeout).
# same_tool_failure (DIFFERENT commands) never halts these; only an exact-args replay with
# no intervening change, or an identical-result streak, can.
FAILURE_TOLERANT_TOOL_NAMES = frozenset({
    "terminal", "execute_code", "process_manage", "process", "browser_navigate", "web_extract",
})

# A successful call to one of these marks progress for every failing signature still counted
# this turn: the next retry is a new experiment (edit -> re-run), not a replay.
PROGRESS_RESET_TOOL_NAMES = frozenset({
    "write_file", "patch", "terminal", "execute_code", "browser_click", "browser_type", "browser_press",
    "browser_navigate", "process_manage", "process", "delegate_task", "send_message", "cronjob",
    "cronjob_manage", "todo", "todo_list", "memory", "skill_manage",
})

_BOOL_FIELDS = ("warnings_enabled", "hard_stop_enabled", "non_interactive_hard_stop_enabled", "session_governor_enabled")
# Threshold field -> (nested section, nested key). The flat legacy key is the field name itself.
_THRESHOLD_SOURCES: dict[str, tuple[str, str]] = {
    "exact_failure_warn_after": ("warn_after", "exact_failure"),
    "same_tool_failure_warn_after": ("warn_after", "same_tool_failure"),
    "no_progress_warn_after": ("warn_after", "idempotent_no_progress"),
    "exact_failure_block_after": ("hard_stop_after", "exact_failure"),
    "same_tool_failure_halt_after": ("hard_stop_after", "same_tool_failure"),
    "no_progress_block_after": ("hard_stop_after", "idempotent_no_progress"),
    # Session-scope Repeat Governor (R1): lives under its own nested section so the
    # per-turn ``warn_after`` keys never collide. ``0`` disables a tier.
    "session_warn_after": ("session", "warn_after"),
    "session_block_after": ("session", "block_after"),
}

# Defaults for the session-scope Repeat Governor (R1): the notice tier fires on the
# 2nd cross-turn identical call, the block tier on the 3rd — mirroring the per-turn
# no-progress ladder (warn 2 / block 5 would over-block legitimate cross-turn
# re-reads; the session scope is strictly more aggressive, so start softer).
_SESSION_WARN_AFTER_DEFAULT = 2
_SESSION_BLOCK_AFTER_DEFAULT = 3
# Bounded ring buffer (review change 2a): stale pointers age out instead of growing
# unbounded across a 24h session.
SESSION_GOVERNOR_BUFFER_SIZE = 200

# Failure-class enum for the session governor's failure hashing (review change 4):
# payloads that embed mutating counters/timestamps must hash identically, so the
# class — not the text — is the key. ``other`` still counts (it only ever widens).
TOOL_FAILURE_CLASSES = frozenset({"auth", "not_found", "capacity", "validation", "timeout", "other"})

# R2 — in-turn wait budget (review: two tiers, agent layer, NOT inside wait()).
# ~100s cumulative: warn ("end the turn; you'll be notified"); ~150s: refuse
# further wait-style calls with the notify/wake contract. A single cliff at 150
# would punish one legitimate long wait (review change 2).
TOOL_WAIT_BUDGET_WARN_SECONDS = 100
TOOL_WAIT_BUDGET_REFUSE_SECONDS = 150

_WAIT_BUDGET_WARN_MESSAGE = (
    "[hermes note: this turn has already spent {seconds}s waiting on background processes "
    "(budget {budget}s). End the turn instead of waiting again — with notify_on_complete you "
    "will be notified on exit; do more work or finish your reply now.]"
)

_WAIT_BUDGET_REFUSE_MESSAGE = (
    "Blocked wait: this turn has already spent {seconds}s waiting on background processes "
    "(budget {budget}s). Do not wait again this turn — the process is still running and you "
    "will be notified on exit (notify_on_complete / completion notification). Do more work or "
    "end the turn now; the wake arrives on its own."
)

_FAILURE_CLASS_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("auth", ("unauthorized", "401", "invalid api key", "authentication", "permission", "forbidden", "403")),
    ("not_found", ("not found", "no such", "404", "does not exist", "unknown action", "unknown tool")),
    ("capacity", ("exceed the limit", "limit reached", "too large", "quota", "429", "rate limit", "exceed")),
    ("timeout", ("timed out", "timeout", "deadline exceeded")),
    ("validation", ("is required", "invalid", "must be", "cannot", "expected", "400", "validation")),
)

# Per-turn caps on runaway-prone tools (counters reset in reset_for_turn).
_DEFAULT_MAX_WEB_SEARCHES_PER_TURN = 50
_DEFAULT_MAX_SUBAGENTS_PER_TURN = 50

# Interactive surfaces plus bounded supervised task loops (subagent stopped by its parent;
# api_server has a live client) doing real edit -> re-run work keep the warn-only default.
_ATTENDED_PLATFORMS = frozenset({"cli", "tui", "desktop", "acp", "subagent", "api_server"})


def is_stall_guard_repeatable(tool_name: str) -> bool:
    """Whether a tool is exempt from the identical-call loop notice."""
    return tool_name in STALL_GUARD_REPEATABLE_TOOLS or tool_name.endswith(_STALL_GUARD_REPEATABLE_SUFFIXES)


def _is_wait_style_call(tool_name: str, args: Mapping[str, Any]) -> bool:
    """True for calls that BLOCK waiting on a background process (R2 budget):
    ``process_manage action=wait``. Polls/logs are cheap observations, not waits."""
    if tool_name not in STALL_GUARD_REPEATABLE_TOOLS:
        return False
    return str(args.get("action") or "").strip().lower() == "wait"


def _is_non_interactive_platform(platform: str | None) -> bool:
    """True for gateway/cron sessions where tool loops are unattended."""
    if not isinstance(platform, str) or not platform.strip():
        return False
    return platform.strip().lower() not in _ATTENDED_PLATFORMS


@dataclass(frozen=True)
class LoopCapConfig:
    """Per-turn hard ceilings on web_search calls / subagent spawns; count total calls (not
    repeats), fire regardless of ``hard_stop_enabled``; ``0`` disables a cap."""

    max_web_searches: int = _DEFAULT_MAX_WEB_SEARCHES_PER_TURN
    max_subagents: int = _DEFAULT_MAX_SUBAGENTS_PER_TURN

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> LoopCapConfig:
        """Build config from the ``tool_loop_guardrails.loop_caps`` section."""
        if not isinstance(data, Mapping):
            return cls()
        return cls(**{f.name: _int_at_least(data.get(f.name), f.default, 0) for f in fields(cls)})


@dataclass(frozen=True)
class ToolCallGuardrailConfig:
    """Thresholds for per-turn tool-call loop detection. Warnings never prevent execution; hard
    stops are opt-in on interactive platforms, default on for unattended gateway/cron platforms."""

    warnings_enabled: bool = True
    hard_stop_enabled: bool = False
    non_interactive_hard_stop_enabled: bool = True
    exact_failure_warn_after: int = 2
    exact_failure_block_after: int = 5
    same_tool_failure_warn_after: int = 3
    same_tool_failure_halt_after: int = 8
    no_progress_warn_after: int = 2
    no_progress_block_after: int = 5
    idempotent_tools: frozenset[str] = field(default_factory=lambda: IDEMPOTENT_TOOL_NAMES)
    mutating_tools: frozenset[str] = field(default_factory=lambda: MUTATING_TOOL_NAMES)
    loop_caps: LoopCapConfig = field(default_factory=LoopCapConfig)
    # Session-scope Repeat Governor (R1): master switch + cross-turn thresholds.
    # ``0`` disables a tier (via _int_at_least minimum=0); the master switch kills
    # every session decision without touching the per-turn detectors.
    session_governor_enabled: bool = True
    session_warn_after: int = _SESSION_WARN_AFTER_DEFAULT
    session_block_after: int = _SESSION_BLOCK_AFTER_DEFAULT

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None, *, platform: str | None = None,
    ) -> ToolCallGuardrailConfig:
        """Build config from `tool_loop_guardrails`; nested ``warn_after`` / ``hard_stop_after`` win over flat legacy keys."""
        if not isinstance(data, Mapping):
            data = {}
        d = cls()
        flags = {name: _as_bool(data.get(name), getattr(d, name)) for name in _BOOL_FIELDS}
        if flags["non_interactive_hard_stop_enabled"] and _is_non_interactive_platform(platform):
            flags["hard_stop_enabled"] = True

        def threshold(name: str, section_name: str, key: str, *, minimum: int = 1) -> int:
            section = data.get(section_name)
            nested = section.get(key, data.get(name)) if isinstance(section, Mapping) else data.get(name)
            # Session governor tiers honor explicit 0 = disabled (review change 7);
            # per-turn thresholds keep minimum 1 (0 would mean block on first call).
            return _int_at_least(nested, getattr(d, name), 0 if name.startswith("session_") else 1)

        thresholds = {name: threshold(name, *src) for name, src in _THRESHOLD_SOURCES.items()}
        return cls(loop_caps=LoopCapConfig.from_mapping(data.get("loop_caps")), **flags, **thresholds)


@dataclass(frozen=True)
class IdenticalCallObservation:
    """``notice`` is appended after the result, ``stub`` replaces a byte-identical duplicate result;
    ``kind`` names the detector behind the notice (``identical_call_streak`` / ``identical_cycle``)."""

    notice: str | None = None
    stub: str | None = None
    kind: str | None = None


@dataclass(frozen=True)
class ToolCallSignature:
    """Stable, non-reversible identity for a tool name plus canonical args."""

    tool_name: str
    args_hash: str

    @classmethod
    def from_call(cls, tool_name: str, args: Mapping[str, Any] | None) -> ToolCallSignature:
        return cls(tool_name=tool_name, args_hash=_sha256(canonical_tool_args(args or {})))

    def to_metadata(self) -> dict[str, str]:
        """Return public metadata without raw argument values."""
        return asdict(self)


@dataclass(frozen=True)
class ToolGuardrailDecision:
    """Decision returned by the tool-call guardrail controller."""

    action: str = "allow"  # allow | warn | block | halt
    code: str = "allow"
    message: str = ""
    tool_name: str = ""
    count: int = 0
    signature: ToolCallSignature | None = None

    @property
    def allows_execution(self) -> bool:
        return self.action in {"allow", "warn"}

    @property
    def should_halt(self) -> bool:
        return self.action in {"block", "halt"}

    def to_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        if data["signature"] is None:
            del data["signature"]
        return data


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def canonical_tool_args(args: Mapping[str, Any]) -> str:
    """Return sorted compact JSON for parsed tool arguments."""
    if not isinstance(args, Mapping):
        raise TypeError(f"tool args must be a mapping, got {type(args).__name__}")
    return _canonical_json(args)


def classify_tool_failure(tool_name: str, result: str | None) -> tuple[bool, str]:
    """Fallback classifier used only when callers don't pass ``failed``; mirrors
    ``agent.display._detect_tool_failure`` so the guardrail never disagrees with the CLI's ``[error]`` tag."""
    if result is None or file_mutation_result_landed(tool_name, result):
        return False, ""

    # A harness REFUSAL of a redundant call (repeated identical read/search) carries
    # ``"error"`` for the model's benefit -- exactly what the substring test below keys
    # on -- but nothing failed; counting it lets the cheap refusal feed the streak that
    # fires the next, harder one. Mirrored in ``agent.display._detect_tool_failure``.
    if is_guardrail_refusal(result):
        return False, ""

    if tool_name == "terminal":
        data = safe_json_loads(result)
        exit_code = data.get("exit_code") if isinstance(data, dict) else None
        return (True, f" [exit {exit_code}]") if exit_code is not None and exit_code != 0 else (False, "")

    if tool_name == "memory":
        data = safe_json_loads(result)
        if isinstance(data, dict) and data.get("success") is False and "exceed the limit" in data.get("error", ""):
            return True, " [full]"
    lower = result[:500].lower()
    return (True, " [error]") if '"error"' in lower or '"failed"' in lower or result.startswith("Error") else (False, "")


# Guardrail verdict text injected into the conversation, keyed by decision code.
# ``same_tool_failure_warning`` is built by _tool_failure_recovery_hint (tool-specific).
_DECISION_MESSAGES: dict[str, str] = {
    "repeated_exact_failure_block": (
        "Blocked {tool_name}: the same tool call failed {count} times with identical arguments. "
        "Stop retrying it unchanged; change strategy or explain the blocker."
    ),
    "idempotent_no_progress_block": (
        "Blocked {tool_name}: this read-only call returned the same result {count} times. "
        "Stop repeating it unchanged; use the result already provided or try a different query."
    ),
    "same_tool_failure_halt": (
        "Stopped {tool_name}: it failed {count} times this turn. "
        "Stop retrying the same failing tool path and choose a different approach."
    ),
    "repeated_exact_failure_warning": (
        "{tool_name} has failed {count} times with identical arguments. This looks like a loop; "
        "inspect the error and change strategy instead of retrying it unchanged."
    ),
    "idempotent_no_progress_warning": (
        "{tool_name} returned the same result {count} times. Use the result already provided "
        "or change the query instead of repeating it unchanged."
    ),
    "identical_call_streak_halt": (
        "Stopped {tool_name}: the same call with identical arguments returned the same result "
        "{count} times in a row. Stop repeating it unchanged; use the result already provided or change strategy."
    ),
    "identical_cycle_halt": (
        "Stopped {tool_name}: the same repeating cycle of tool calls (period {period}) with identical "
        "arguments and identical results has run {count} times. Repeating the batch unchanged is not "
        "progress; use the results already provided or change strategy."
    ),
    "loop_web_search_cap": (
        "Blocked web_search: this turn has already made {cap} web searches, the per-turn limit. "
        "This looks like a runaway search loop. Work with the results you already have and give the user your answer."
    ),
    "loop_subagent_cap": (
        "Blocked delegate_task: this turn has already spawned {count} subagents (limit {cap}). "
        "This looks like a runaway delegation loop. Finish the work with the results you have and answer the user."
    ),
}

_IDENTICAL_CALL_NOTICE = (
    "[hermes note: this is the {ordinal} consecutive identical call to "
    "{tool_name} with identical arguments returning the same result. "
    "Do not repeat it — change arguments, use a different tool, or "
    "proceed with what you have.]"
)

_IDENTICAL_CYCLE_NOTICE = (
    "[hermes note: the last {count} rounds repeated the same cycle of {period} tool calls "
    "(ending with {tool_name}) with identical arguments and identical results. "
    "Do not repeat the batch — change arguments, use a different tool, or "
    "proceed with what you have.]"
)

# tool -> (LoopCapConfig field, controller counter attribute, decision code)
_LOOP_CAPS: dict[str, tuple[str, str, str]] = {
    "web_search": ("max_web_searches", "_turn_web_search_count", "loop_web_search_cap"),
    "delegate_task": ("max_subagents", "_turn_subagent_count", "loop_subagent_cap"),
}


class ToolCallGuardrailController:
    """Per-turn controller for repeated failed/non-progressing tool calls."""

    def __init__(self, config: ToolCallGuardrailConfig | None = None):
        self.config = config or ToolCallGuardrailConfig()
        self._init_session_state()
        self.reset_for_turn()

    def _init_session_state(self) -> None:
        """Session-scope Repeat Governor state (R1) — lives OUTSIDE reset_for_turn()
        so cross-turn loops stay visible; ``reset_for_session()`` clears it at
        agent init / new session."""
        # Ring buffer of (signature, result_hash_or_failure_class, failed, turn_epoch).
        # Bounded (SESSION_GOVERNOR_BUFFER_SIZE): stale entries age out so a compaction
        # can't leave dangling pointers living forever (review change 2a).
        self._session_buffer: deque[tuple[ToolCallSignature, str, bool, int]] = deque(maxlen=SESSION_GOVERNOR_BUFFER_SIZE)
        # Per-turn epoch counter: the de-overlap gate (review change 1). A session
        # decision fires only when the PREVIOUS identical call happened in an
        # EARLIER turn — within a turn, the existing per-turn stub/notice stands.
        self._session_turn_epoch = 0
        # signature -> last turn_epoch it was seen in (for the de-overlap gate).
        self._session_last_seen: dict[ToolCallSignature, int] = {}
        # signature -> cross-turn identical-call count (success AND failure paths).
        self._session_counts: dict[ToolCallSignature, int] = {}
        # signature -> the FIRST observation's result preview/failure-class, so the
        # notice/block can point at the earlier result (review R1 design).
        self._session_first_result: dict[ToolCallSignature, str] = {}
        # signature -> turn epoch of the last successful PROGRESS_RESET call; mirrors
        # the per-turn _progress_since_failure, session-scoped (review change 3).
        self._session_progress_epoch: dict[ToolCallSignature, int] = {}
        # Session-scoped persisted-path pointers (A1): a stub referencing a spillover
        # file must not dangle just because the turn ended.
        self._persisted_result_paths_session: dict[str, str] = {}
        # signature -> the tool_call_id of the FIRST observed call (review round 1,
        # defect 2): the session-governor notice/block cites the spillover file for
        # THIS signature's own first result, never an arbitrary other call's file.
        self._session_first_call_id: dict[ToolCallSignature, str] = {}
        # Parked block decisions from session_observe: signature -> decision, consumed
        # by the runtime's before-call gate on the NEXT identical call.
        self._session_block_pending: dict[ToolCallSignature, ToolGuardrailDecision] = {}

    def reset_for_session(self) -> None:
        """Clear ALL state including the session-scope governor. Called at agent
        init / new session — never per turn."""
        self._init_session_state()
        self.reset_for_turn()

    def reset_for_turn(self) -> None:
        self._exact_failure_counts: dict[ToolCallSignature, int] = {}
        self._same_tool_failure_counts: dict[str, int] = {}
        # signature -> a mutating call succeeded since its last failure
        self._progress_since_failure: dict[ToolCallSignature, bool] = {}
        self._no_progress: dict[ToolCallSignature, tuple[str, int]] = {}
        self._halt_decision: ToolGuardrailDecision | None = None
        # Identical-call streak: CONSECUTIVE identical (tool, args, result) calls; any different call or
        # result resets it, so re-reads after edits and varied polling are never flagged.
        # Identical-call loop-breaker state (agent.stall_guards): tracks the CONSECUTIVE streak of identical
        # (tool, canonical args) calls whose results were also identical. Per-turn, like everything else
        # here. NOTE: open PR #85352 (patrykkopycinski) tracks no-progress loops ACROSS turns via a
        # detection window — a different mechanism from this per-turn consecutive streak. Coordinate future
        # work there.
        self._identical_streak_sig: ToolCallSignature | None = None
        self._identical_streak_result_hash: str = ""
        self._identical_streak_count: int = 0
        self._identical_streak_first_call_id: str = ""
        # Batch-cycle loop breaker (port of can1357/oh-my-pi#10521): sequence of
        # (signature, result_hash, repeatable) for every observed call this turn, so a repeating
        # multi-call cycle (A,B,A,B,...) is caught even though it resets the consecutive streak above.
        self._call_history: deque[tuple[ToolCallSignature, str, bool]] = deque(maxlen=_STALL_GUARD_CYCLE_HISTORY)
        # tool_call_id -> spillover path, so a stub referencing a persisted-output preview can't dangle.
        # Session-scoped as of R1/A1 (the governor points at these files across turns);
        # the per-turn map below is kept as the alias ``_persisted_result_paths`` reads/writes go through.
        self._persisted_result_paths = self._persisted_result_paths_session
        self._turn_web_search_count = 0
        self._turn_subagent_count = 0
        # R2 wait budget: cumulative seconds this turn spent in wait-style calls.
        self._turn_wait_seconds = 0.0
        # R2: parked refusal for the NEXT wait call (the current one already ran).
        self._wait_refusal_pending = False
        # The turn epoch advances at reset_for_turn (every turn start / between the
        # caller's turns): the session governor's de-overlap gate compares epochs.
        self._session_turn_epoch += 1

    @property
    def halt_decision(self) -> ToolGuardrailDecision | None:
        return self._halt_decision

    def _decide(
        self, action: str, code: str, tool_name: str, count: int, signature: ToolCallSignature,
        *, message: str | None = None, **fmt: Any,
    ) -> ToolGuardrailDecision:
        """Build a warn/block/halt decision; block/halt is also recorded as the turn's halt decision."""
        if message is None:
            message = _DECISION_MESSAGES[code].format(tool_name=tool_name, count=count, **fmt)
        decision = ToolGuardrailDecision(action, code, message, tool_name, count, signature)
        if decision.should_halt:
            self._halt_decision = decision
        return decision

    def before_call(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolGuardrailDecision:
        args = _coerce_args(args)
        signature = ToolCallSignature.from_call(tool_name, args)
        allow = ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        # Loop caps apply regardless of hard_stop_enabled (which only governs the detector).
        cap_block = self._check_loop_cap(tool_name, args, signature)
        if cap_block is not None or not self.config.hard_stop_enabled:
            return cap_block or allow
        # A mutation since this call last failed makes the retry a new experiment.
        exact_count = 0 if self._progress_since_failure.get(signature) else self._exact_failure_counts.get(signature, 0)
        if exact_count >= self.config.exact_failure_block_after:
            return self._decide("block", "repeated_exact_failure_block", tool_name, exact_count, signature)
        record = self._no_progress.get(signature) if self._is_idempotent(tool_name) else None
        if record is not None and record[1] >= self.config.no_progress_block_after:
            return self._decide("block", "idempotent_no_progress_block", tool_name, record[1], signature)
        return allow

    def after_call(
        self, tool_name: str, args: Mapping[str, Any] | None, result: str | None,
        *, failed: bool | None = None,
    ) -> ToolGuardrailDecision:
        args = _coerce_args(args)
        signature = ToolCallSignature.from_call(tool_name, args)
        if failed is None:
            failed, _ = classify_tool_failure(tool_name, result)
        warnings = self.config.warnings_enabled

        if failed:
            # An identical failing call is only a REPLAY if nothing landed in between;
            # a mutation since the last identical failure restarts the exact-args streak.
            if self._progress_since_failure.pop(signature, False):
                self._exact_failure_counts.pop(signature, None)
            exact_count = self._exact_failure_counts[signature] = self._exact_failure_counts.get(signature, 0) + 1
            same_count = self._same_tool_failure_counts[tool_name] = self._same_tool_failure_counts.get(tool_name, 0) + 1
            self._no_progress.pop(signature, None)
            # same_tool_failure counts DIFFERENT args on one tool; for failure-tolerant
            # tools a run of distinct red commands is diagnosis, not a loop — warn, never halt.
            if (
                # Hard-stop widening (#89069 / #100849 bundle): the per-turn no-progress BLOCK above only
                # covers tools in idempotent_tools, so a model replaying the same successful
                # `terminal`/`skill_view` call with a byte-identical result ran until the iteration budget.
                # The consecutive-identical streak is tool-agnostic; when hard stops are enabled, halt at
                # the same idempotent_no_progress threshold. Pollers stay exempt (an unchanged poll is
                # progress).
                self.config.hard_stop_enabled
                and tool_name not in FAILURE_TOLERANT_TOOL_NAMES
                and same_count >= self.config.same_tool_failure_halt_after
            ):
                return self._decide("halt", "same_tool_failure_halt", tool_name, same_count, signature)
            if warnings and exact_count >= self.config.exact_failure_warn_after:
                return self._decide("warn", "repeated_exact_failure_warning", tool_name, exact_count, signature)
            if warnings and same_count >= self.config.same_tool_failure_warn_after:
                return self._decide(
                    "warn", "same_tool_failure_warning", tool_name, same_count, signature,
                    message=_tool_failure_recovery_hint(tool_name, same_count),
                )
            return ToolGuardrailDecision(tool_name=tool_name, count=exact_count, signature=signature)

        self._exact_failure_counts.pop(signature, None)
        self._same_tool_failure_counts.pop(tool_name, None)
        # A successful mutation is progress for every failing signature still counted
        # this turn. Pure loops never mutate between attempts, so the replay detector keeps its teeth.
        if tool_name in PROGRESS_RESET_TOOL_NAMES or file_mutation_result_landed(tool_name, result):
            self._progress_since_failure.update(dict.fromkeys(self._exact_failure_counts, True))
            self._same_tool_failure_counts.clear()
        if not self._is_idempotent(tool_name):
            self._no_progress.pop(signature, None)
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        result_hash = _result_hash(result, tool_name)
        previous = self._no_progress.get(signature)
        repeat_count = previous[1] + 1 if previous is not None and previous[0] == result_hash else 1
        self._no_progress[signature] = (result_hash, repeat_count)
        if warnings and repeat_count >= self.config.no_progress_warn_after:
            return self._decide("warn", "idempotent_no_progress_warning", tool_name, repeat_count, signature)
        return ToolGuardrailDecision(tool_name=tool_name, count=repeat_count, signature=signature)

    def _is_idempotent(self, tool_name: str) -> bool:
        return tool_name not in self.config.mutating_tools and tool_name in self.config.idempotent_tools

    def observe_call(
        self, tool_name: str, args: Mapping[str, Any] | None, result: str | None,
        *, tool_call_id: str = "", failed: bool = False,
    ) -> IdenticalCallObservation:
        """Track consecutive identical calls; return notice + dedupe stub info.

        ``notice`` fires from the threshold-th consecutive identical (tool, args, result) call
        (observational, pollers exempt). ``stub`` replaces the CURRENT result from the 2nd byte-identical
        repeat — the tool still executed, only the context representation is deduplicated, so polling
        semantics survive; pollers are NOT exempt here since an unchanged poll is where it saves most.
        """
        is_plain_str = isinstance(result, str)
        signature = ToolCallSignature.from_call(tool_name, _coerce_args(args))
        result_hash = _result_hash(result, tool_name) if is_plain_str else ""

        if is_plain_str and (signature, result_hash) == (self._identical_streak_sig, self._identical_streak_result_hash):
            self._identical_streak_count += 1
        else:
            # New streak; non-string (multimodal) results never form one.
            self._identical_streak_sig = signature if is_plain_str else None
            self._identical_streak_result_hash = result_hash
            self._identical_streak_count = 1 if is_plain_str else 0
            self._identical_streak_first_call_id = tool_call_id or ""
        count = self._identical_streak_count

        notice = kind = None
        if not is_stall_guard_repeatable(tool_name) and count >= STALL_GUARD_IDENTICAL_CALL_THRESHOLD:
            notice, kind = _IDENTICAL_CALL_NOTICE.format(ordinal=_ordinal(count), tool_name=tool_name), "identical_call_streak"
            # The no-progress BLOCK in before_call only covers idempotent_tools; this streak
            # is tool-agnostic, so with hard stops on, halt at the same threshold (a model
            # replaying a successful `terminal` call otherwise runs to the budget).
            if self.config.hard_stop_enabled and count >= self.config.no_progress_block_after and self._halt_decision is None:
                self._decide("halt", "identical_call_streak_halt", tool_name, count, signature)

        # Batch-cycle detection (oh-my-pi#10521): a repeating multi-call cycle resets the
        # consecutive streak on every alternation, so check the call history for a period-p lap.
        if is_plain_str:
            self._call_history.append((signature, result_hash, is_stall_guard_repeatable(tool_name)))
        else:
            self._call_history.clear()
        if notice is None and is_plain_str:
            cycle = self._detect_identical_cycle()
            if cycle is not None:
                period, laps = cycle
                notice, kind = _IDENTICAL_CYCLE_NOTICE.format(count=laps, period=period, tool_name=tool_name), "identical_cycle"
                if self.config.hard_stop_enabled and laps >= self.config.no_progress_block_after and self._halt_decision is None:
                    self._decide("halt", "identical_cycle_halt", tool_name, laps, signature, period=period)

        stub = None
        if is_plain_str and count >= 2 and not failed and len(result) >= IDENTICAL_RESULT_STUB_MIN_CHARS:
            stub = self._build_result_reference_stub(tool_name, args)
        return IdenticalCallObservation(notice=notice, stub=stub, kind=kind)

    def _detect_identical_cycle(self) -> tuple[int, int] | None:
        """Detect a repeating identical-call cycle ending at the latest observed call.

        Returns ``(period, laps)`` for the smallest period 2..max whose trailing laps
        (identical signature AND result per position) reach the notice threshold, else None.
        Period 1 is the consecutive streak's job. A cycle made ONLY of poller-exempt tools
        is exempt (an unchanged poll loop is legitimate waiting); one non-exempt call in
        the cycle keeps the guard armed, matching the single-call exemption semantics.
        """
        history = self._call_history
        for period in range(2, _STALL_GUARD_MAX_CYCLE_PERIOD + 1):
            if len(history) < period * STALL_GUARD_IDENTICAL_CALL_THRESHOLD:
                continue
            laps = 1
            # Count how many consecutive trailing laps equal the final lap.
            while True:
                base = len(history) - period * (laps + 1)
                if base < 0:
                    break
                lap_equal = all(
                    history[base + i][:2] == history[len(history) - period + i][:2]
                    for i in range(period)
                )
                if not lap_equal:
                    break
                laps += 1
            if laps >= STALL_GUARD_IDENTICAL_CALL_THRESHOLD:
                tail = [history[len(history) - period + i] for i in range(period)]
                if all(repeatable for _, _, repeatable in tail):
                    continue
                # A constant sub-cycle would already have fired at a smaller period.
                return period, laps
        return None

    # ── Session-scope Repeat Governor (R1) ────────────────────────────────────

    def session_observe(
        self, tool_name: str, args: Mapping[str, Any] | None, result: Any,
        *, failed: bool = False, tool_call_id: str = "",
    ) -> ToolGuardrailDecision:
        """Cross-turn repeat detector: the per-turn detectors are wiped by
        ``reset_for_turn()`` at every turn start, so 83% of red's repeats (multi-hour
        loops across turns) were invisible. This tracks (tool, args-hash,
        result-hash | failure-class) in a session-scoped ring buffer and returns
        warn/block decisions on the ``session.warn_after`` / ``session.block_after``
        thresholds.

        Gates (review changes 1-7):
        - De-overlap: fires only when the previous identical call happened in an
          EARLIER turn (turn-epoch compare) — within a turn the per-turn stub/notice
          already handles it.
        - Progress reset: a successful call to a PROGRESS_RESET_TOOL_NAMES tool
          after the signature's last failure makes the next retry a new experiment.
        - Failure-class hashing: ``classify_tool_failure`` stays the failure
          detector; the session key normalizes the payload to a class so mutating
          counters/timestamps ("failed 4 times" -> "5 times") can't defeat the hash.
        - Pollers are exempt from the SUCCESS-repeat tier ONLY: an identical
          failing poller result still counts (12 wait timeouts -> re-poll).
        - Count-before-gate: the observation is recorded BEFORE the block decision
          is built, so the first blocked call is never unrecorded (A3).
        """
        allow = ToolGuardrailDecision(tool_name=tool_name, signature=ToolCallSignature.from_call(tool_name, _coerce_args(args)))
        if not self.config.session_governor_enabled:
            return allow
        signature = allow.signature
        # Progress reset (review change 3): a successful call to a PROGRESS_RESET
        # tool marks progress for every signature still counted — stamped with the
        # CURRENT epoch so the check below can compare "reset happened after the
        # previous identical failure".
        if not failed and tool_name in PROGRESS_RESET_TOOL_NAMES:
            for counted_sig in self._session_counts:
                self._session_progress_epoch[counted_sig] = self._session_turn_epoch
        if failed:
            failure_class = _failure_class(tool_name, result if isinstance(result, str) else "")
            value = f"fail:{failure_class}"
        else:
            # A2: non-string (multimodal) results hash instead of being exempt.
            value = f"ok:{_result_hash(result if isinstance(result, str) else _canonical_json(result))}"

        epoch_now = self._session_turn_epoch
        previous_epoch = self._session_last_seen.get(signature)
        previous_value = self._session_first_result.get(signature)

        # Record BEFORE deciding (A3): a blocked repeat still advances the count.
        self._session_buffer.append((signature, value, failed, epoch_now))
        self._session_last_seen[signature] = epoch_now
        if previous_value is None:
            self._session_first_result[signature] = value
            # First sighting: remember this call's id so the repeat message can cite
            # the signature's OWN spillover file (review round 1, defect 2). The
            # spillover is recorded by the runtime right after this observe (the
            # persist step follows the guardrail observation), and the pointer map
            # is session-scoped (A1) — so the path resolves across turns.
            self._session_first_call_id[signature] = tool_call_id or ""

        if previous_epoch is None or previous_value != value:
            # First sighting this session, or the result/class changed — no repeat.
            self._session_counts.pop(signature, None)
            return allow

        # De-overlap gate (review change 1): the previous identical call must have
        # happened in an EARLIER turn. Within one turn the existing per-turn
        # stub/notice handles the repeat; the session governor stays out.
        if previous_epoch >= epoch_now:
            return allow

        if self._session_progress_epoch.get(signature, -1) >= previous_epoch and (
            failed or tool_name not in PROGRESS_RESET_TOOL_NAMES
        ):
            # A mutation succeeded after the previous identical call: the retry is a
            # new experiment (review change 3), restart the count. Applies to the
            # failure tier unconditionally and to the success tier when THIS call is
            # not itself the progress-resetting mutation.
            self._session_counts.pop(signature, None)
            self._session_first_result[signature] = value
            return allow

        count = self._session_counts[signature] = self._session_counts.get(signature, 0) + 1

        # Poller exemption, SUCCESS tier only (review change 6): an unchanged poll
        # is legitimate waiting; an unchanged FAILURE still counts.
        if not failed and is_stall_guard_repeatable(tool_name):
            return allow

        if self.config.session_block_after and count + 1 >= self.config.session_block_after:
            # The current call already executed — the block prevents the NEXT
            # identical one. Park it for the runtime's before-call gate (NOT
            # _decide: that would set _halt_decision and misreport the completed
            # call as a turn halt).
            decision = ToolGuardrailDecision(
                action="block", code="session_repeat_block", tool_name=tool_name, count=count + 1,
                signature=signature,
                message=_session_repeat_message(
                    tool_name, count + 1, failed, previous_value,
                    self._persisted_result_paths,
                    first_call_id=self._session_first_call_id.get(signature, ""),
                ),
            )
            self._session_block_pending[signature] = decision
            return decision
        if self.config.session_warn_after and count + 1 >= self.config.session_warn_after:
            return self._decide(
                "warn", "session_repeat_notice", tool_name, count + 1, signature,
                message=_session_repeat_message(
                    tool_name, count + 1, failed, previous_value,
                    self._persisted_result_paths,
                    first_call_id=self._session_first_call_id.get(signature, ""),
                ),
            )
        return allow

    def session_block_for(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolGuardrailDecision | None:
        """The pending session-governor BLOCK for this exact call, if any.

        The governor observes post-call (it needs the result), but its block must
        prevent the NEXT execution — so a block decision is parked here and
        ``before_call`` consults this before allowing. Mirrors
        ``_check_loop_cap``'s count-before-gate ordering (A3)."""
        if not self.config.session_governor_enabled:
            return None
        signature = ToolCallSignature.from_call(tool_name, _coerce_args(args))
        return self._session_block_pending.get(signature)

    def session_before_call(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolGuardrailDecision | None:
        """Consume the pending session block for this call (called by the runtime's
        before-call gate). Returns the parked decision exactly once."""
        if not self.config.session_governor_enabled:
            return None
        signature = ToolCallSignature.from_call(tool_name, _coerce_args(args))
        return self._session_block_pending.pop(signature, None)

    # ── R2: agent-layer in-turn wait budget ───────────────────────────────────

    def wait_budget_block_for(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolGuardrailDecision | None:
        """The parked R2 refusal for a wait-style call about to dispatch, if any."""
        if not self._wait_refusal_pending:
            return None
        if not _is_wait_style_call(tool_name, _coerce_args(args)):
            return None
        return ToolGuardrailDecision(
            action="block", code="wait_budget_refused", tool_name=tool_name,
            count=int(self._turn_wait_seconds), signature=ToolCallSignature.from_call(tool_name, _coerce_args(args)),
            message=_WAIT_BUDGET_REFUSE_MESSAGE.format(
                seconds=int(self._turn_wait_seconds), budget=TOOL_WAIT_BUDGET_REFUSE_SECONDS,
            ),
        )

    def observe_wait_window(self, seconds: float, *, failed: bool = False) -> ToolGuardrailDecision | None:
        """Record a completed wait-style call's wall-clock window against this
        turn's budget (R2). Called from the runtime's post-call observation path —
        NOT from ``process_registry.wait()``, which has no turn context. The wait
        itself always runs; past the REFUSE tier the NEXT wait is refused
        (parked decision consumed by the before-call gate).

        Poller wait failures still count — the 12 timeout→re-poll rounds are the
        pattern this budget exists to break (review: poller failures not exempt).
        Returns a warn/block decision to append, or None."""
        self._turn_wait_seconds += max(0.0, float(seconds or 0))
        if self._turn_wait_seconds >= TOOL_WAIT_BUDGET_REFUSE_SECONDS:
            # Park the refusal for the next wait-style call; the current result
            # carries the block message so the model reads the contract NOW.
            self._wait_refusal_pending = True
            return ToolGuardrailDecision(
                action="block", code="wait_budget_refused", tool_name="process_manage",
                count=int(self._turn_wait_seconds), signature=None,
                message=_WAIT_BUDGET_REFUSE_MESSAGE.format(
                    seconds=int(self._turn_wait_seconds), budget=TOOL_WAIT_BUDGET_REFUSE_SECONDS,
                ),
            )
        if self._turn_wait_seconds >= TOOL_WAIT_BUDGET_WARN_SECONDS:
            return ToolGuardrailDecision(
                action="warn", code="wait_budget_warning", tool_name="process_manage",
                count=int(self._turn_wait_seconds), signature=None,
                message=_WAIT_BUDGET_WARN_MESSAGE.format(
                    seconds=int(self._turn_wait_seconds), budget=TOOL_WAIT_BUDGET_WARN_SECONDS,
                ),
            )
        return None

    def record_persisted_result(self, tool_call_id: str, file_path: str) -> None:
        """Remember the spillover path a persisted result was saved to."""
        if tool_call_id and file_path:
            self._persisted_result_paths[tool_call_id] = file_path

    def _build_result_reference_stub(self, tool_name: str, args: Mapping[str, Any] | None) -> str:
        """Reference stub for a byte-identical duplicate result (tool + args preview)."""
        args_preview = canonical_tool_args(_coerce_args(args))
        if len(args_preview) > _RESULT_STUB_ARGS_PREVIEW_CHARS:
            args_preview = args_preview[:_RESULT_STUB_ARGS_PREVIEW_CHARS] + "…"
        first_id = self._identical_streak_first_call_id
        ref = f" (tool_call_id {first_id})" if first_id else ""
        stub = (
            f"[hermes note: this result is byte-identical to the {tool_name} "
            f"result earlier this turn{ref}. Refer to that result; it has not "
            f"changed. Args: {args_preview}]"
        )
        spill_path = self._persisted_result_paths.get(first_id) if first_id else None
        if spill_path:
            stub += f"\n[The referenced result was persisted to: {spill_path} — page through it with read_file if you need the full content.]"
        return stub

    def _check_loop_cap(
        self, tool_name: str, args: Mapping[str, Any], signature: ToolCallSignature,
    ) -> ToolGuardrailDecision | None:
        """Block once a per-turn cap is reached (BEFORE the call, so the (cap+1)-th is refused), else advance
        the counter and return None. delegate_task control actions spawn nothing and keep working after the cap."""
        spec = _LOOP_CAPS.get(tool_name)
        if spec is None:
            return None
        cap_field, count_attr, code = spec
        cap, count = getattr(self.config.loop_caps, cap_field), getattr(self, count_attr)
        increment = 1 if tool_name == "web_search" else (_subagent_spawn_count(args) if cap else 0)
        if increment and cap and count >= cap:
            return self._decide("block", code, tool_name, count, signature, cap=cap)
        setattr(self, count_attr, count + increment)
        return None


def toolguard_synthetic_result(decision: ToolGuardrailDecision) -> str:
    """Build a synthetic role=tool content string for a blocked tool call.

    Carries ``guardrail_refusal: True``: a harness REFUSAL of a redundant call is
    not a failure the tool hit — without the marker the failure classifiers count
    the block and the cheap refusal feeds the very streak that fired it (review
    change 5 / agent.tool_result_classification.GUARDRAIL_REFUSAL_KEY)."""
    return json.dumps(
        {"error": decision.message, "guardrail": decision.to_metadata(),
         GUARDRAIL_REFUSAL_KEY: True},
        ensure_ascii=False,
    )


def append_toolguard_guidance(result: str, decision: ToolGuardrailDecision) -> str:
    """Append runtime guidance to the current tool result content."""
    if decision.action not in {"warn", "halt"} or not decision.message:
        return result
    label = "Tool loop hard stop" if decision.action == "halt" else "Tool loop warning"
    return (result or "") + f"\n\n[{label}: {decision.code}; count={decision.count}; {decision.message}]"


def _tool_failure_recovery_hint(tool_name: str, count: int) -> str:
    """Action-oriented guidance for recovering from repeated tool failures."""
    common = (
        f"{tool_name} has failed {count} times this turn. This looks like a loop. "
        "Do not switch to text-only replies; keep using tools, but diagnose before retrying. "
        "First inspect the latest error/output and verify your assumptions. "
    )
    if tool_name == "terminal":
        return common + (
            "For terminal failures, run a small diagnostic such as `pwd && ls -la` "
            "in the same tool, then try an absolute path, a simpler command, a different "
            "working directory, or a different tool such as read_file/write_file/patch."
        )
    return common + (
        "Try different arguments, a narrower query/path, an absolute path when relevant, "
        "or a different tool that can make progress. If the blocker is external, report "
        "the blocker after one diagnostic attempt instead of repeating the same failing path."
    )


def _ordinal(count: int) -> str:
    return f"{count}{'th' if 11 <= count % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(count % 10, 'th')}"


def _failure_class(tool_name: str, result: str | None) -> str:
    """Failure-class enum for session hashing (review change 4): strips digits/
    timestamps by construction — the class, not the text, is the key. Digits in
    patterns like '401'/'429' are STATUS CLASSES, not counters; a mutating
    counter ("failed 4 times" -> "5 times") maps to the same class either way."""
    if not result:
        return "other"
    lowered = result[:2000].lower()
    # Strip digits that are NOT status codes: counter defeats embed "N times".
    normalized = re.sub(r"\b\d+\b(?= (?:times|attempts|tries))", "N", lowered)
    for failure_class, patterns in _FAILURE_CLASS_PATTERNS:
        if any(p in normalized for p in patterns):
            return failure_class
    return "other"


_SESSION_REPEAT_SUCCESS_MESSAGE = (
    "[hermes note: {tool_name} was already called with these exact arguments earlier in this "
    "session ({ordinal} cross-turn repeat) and returned the same result — the result above is "
    "unchanged. Do not re-run it unchanged; use the earlier result or change the query.]"
)

_SESSION_REPEAT_FAILURE_MESSAGE = (
    "[hermes note: {tool_name} already failed with these exact arguments earlier in this session "
    "({ordinal} cross-turn repeat, failure class: {failure_class}). The identical retry will fail "
    "the same way. Change strategy instead of repeating the call unchanged.]"
)

_SESSION_REPEAT_BLOCK_MESSAGE = (
    "Blocked {tool_name}: this exact call already ran {count} times this session with identical "
    "arguments and {result_phrase}. {change_advice}"
)

_RESULT_PHRASE_SUCCESS = "returned the same result"
_RESULT_PHRASE_FAILURE = "failed with the same error class (see the earlier result above)"

_SESSION_PERSISTED_PATH_NOTE = (
    "\nThe earlier result was persisted to {path} — page through it with read_file if you need the full content."
)


def _session_repeat_message(
    tool_name: str, count: int, failed: bool, previous_value: str | None,
    persisted_paths: Mapping[str, str] | None = None, first_call_id: str = "",
) -> str:
    """Session-governor notice/block text. Tiers share one builder: the notice
    points at the earlier result, the block refuses and cites it."""
    failure_class = previous_value[5:] if previous_value and previous_value.startswith("fail:") else "other"
    if failed:
        change_advice = "Change strategy instead of repeating the call unchanged."
        result_phrase = _RESULT_PHRASE_FAILURE
        text = _SESSION_REPEAT_FAILURE_MESSAGE.format(tool_name=tool_name, ordinal=_ordinal(count), failure_class=failure_class)
    else:
        change_advice = "Use the result already provided or try a different query."
        result_phrase = _RESULT_PHRASE_SUCCESS
        text = _SESSION_REPEAT_SUCCESS_MESSAGE.format(tool_name=tool_name, ordinal=_ordinal(count))
    if count >= 3:
        text = _SESSION_REPEAT_BLOCK_MESSAGE.format(
            tool_name=tool_name, count=count, result_phrase=result_phrase, change_advice=change_advice,
        )
        # Point at the persisted spillover for THIS signature's own first result
        # (dangling-reference mitigation, review change 2b + round-1 defect 2):
        # keyed by the first observed call's tool_call_id — never an arbitrary
        # other call's file. Attached only when that call actually spilled.
        own_path = persisted_paths.get(first_call_id) if (persisted_paths and first_call_id) else None
        if own_path:
            text += _SESSION_PERSISTED_PATH_NOTE.format(path=own_path)
    return text


def _coerce_args(args: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return args if isinstance(args, Mapping) else {}


# execute_code reports per-call kernel bookkeeping — a running `kernel.execution_count` and wall-clock
# `duration_seconds` — that changes on every invocation even when the code did the same thing. Hashed
# as-is, every replay of an identical empty probe looks new and the identical-call streak never forms
# (a model re-ran one empty execute_code call 147 times unflagged). Only these known locations in
# execute_code's own result shape are dropped: for any other tool the same key names can be real output.
def _without_execute_code_metadata(parsed: Any) -> Any:
    if not isinstance(parsed, dict):
        return parsed
    cleaned = {k: v for k, v in parsed.items() if k != "duration_seconds"}
    kernel = cleaned.get("kernel")
    if isinstance(kernel, dict):
        cleaned["kernel"] = {k: v for k, v in kernel.items() if k != "execution_count"}
    return cleaned


def _result_hash(result: str | None, tool_name: str = "") -> str:
    parsed = safe_json_loads(result or "")
    if parsed is None:
        return _sha256(result or "")
    if tool_name == "execute_code":
        parsed = _without_execute_code_metadata(parsed)
    return _sha256(_canonical_json(parsed))


_BOOL_WORDS = {w: True for w in ("1", "true", "yes", "on", "enabled")} | {w: False for w in ("0", "false", "no", "off", "disabled")}


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, (bool, int, float)):
        return bool(value)
    if isinstance(value, str):
        return _BOOL_WORDS.get(value.strip().lower(), default)
    return default


def _int_at_least(value: Any, default: int, minimum: int) -> int:
    """junk/None/below-minimum fall back to default (caps use minimum 0 so 0 = disabled)."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def _subagent_spawn_count(args: Mapping[str, Any]) -> int:
    """Subagents one delegate_task call spawns: ``len(tasks)`` for a non-empty batch, else 1; control actions 0."""
    if str(args.get("action") or "").strip().lower() in ("list", "steer", "stop"):
        return 0
    tasks = args.get("tasks")
    return len(tasks) if isinstance(tasks, list) and tasks else 1


def _sha256(value: str) -> str:
    # surrogatepass: web-scraped results can carry unpaired UTF-16 surrogates; a
    # strict encode would raise and take down the conversation loop.
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()
