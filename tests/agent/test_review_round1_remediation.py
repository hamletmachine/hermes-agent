"""Review round 1 remediation (task t_736d0e16, run 33) — forced-trigger tests.

Defect 1 (R4 config knob dead code): ``agent.silent_streak_checkpoint`` is read
via getattr in agent/silent_streak.py but was never set by any config->agent
path. These tests pin the plumbing: the registered config default
(``hermes_cli.config_defaults.DEFAULT_CONFIG['agent']['silent_streak_checkpoint']``)
and the ``_apply_agent_section`` setattr that makes the override reachable from
every agent-creation path (CLI, cron, gateway — all funnel through init_agent).

Defect 2 (arbitrary persisted-path pointer): ``_session_repeat_message`` picked
``next(iter(persisted_paths.values()))`` — an arbitrary dict-ordered path from
the unpruned tool_call_id-keyed map, NOT the spillover for the repeated
signature. The message must cite the path recorded for the signature's own
first tool_call_id (review change 2b's dangling-reference mitigation, fixed
keyed by the observing call).
"""

import pytest

from agent.tool_guardrails import ToolCallGuardrailController


# ── Defect 1: agent.silent_streak_checkpoint is config-reachable ─────────────


def test_silent_streak_checkpoint_registered_in_config_defaults():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert "silent_streak_checkpoint" in DEFAULT_CONFIG["agent"]


def test_silent_streak_checkpoint_key_validated_as_known():
    from hermes_cli.config import _validate_config_key

    is_known, _ = _validate_config_key("agent.silent_streak_checkpoint")
    assert is_known


def test_apply_agent_section_plumbs_silent_streak_checkpoint():
    """``hermes config set agent.silent_streak_checkpoint N`` must reach the agent."""
    from types import SimpleNamespace

    from agent.agent_init import _apply_agent_section

    agent = SimpleNamespace(run_budget_seconds=None)
    _apply_agent_section(agent, {"agent": {"silent_streak_checkpoint": 42}})
    assert agent.silent_streak_checkpoint == 42


def test_apply_agent_section_defaults_silent_streak_checkpoint_to_none():
    """No config entry -> agent attribute is None -> silent_streak computes it."""
    from types import SimpleNamespace

    from agent.agent_init import _apply_agent_section
    from agent.silent_streak import compute_silent_streak_checkpoint

    agent = SimpleNamespace(run_budget_seconds=None, max_iterations=500)
    _apply_agent_section(agent, {"agent": {}})
    assert agent.silent_streak_checkpoint is None
    assert compute_silent_streak_checkpoint(agent) == 60  # computed, not overridden


# ── Defect 2: persisted-path pointer is the signature's OWN file ─────────────


def _governor_with_spillovers():
    c = ToolCallGuardrailController()
    # Two different calls spilled to different files (tool_call_id -> path).
    c.record_persisted_result("call-a", "/tmp/spill-a.md")
    c.record_persisted_result("call-b", "/tmp/spill-b.md")
    return c


def test_session_repeat_message_cites_own_path_not_arbitrary():
    """The block text for read_file('/a') must point at read_file('/a')'s own
    spillover file, not a random other tool's file."""
    c = _governor_with_spillovers()
    # read_file('/a') observed in an earlier turn with tool_call_id call-a.
    c.observe_call("read_file", {"path": "/a"}, "big result a", tool_call_id="call-a")
    c.record_persisted_result("call-a", "/tmp/spill-a.md")
    c.observe_call("web_search", {"query": "q"}, "search results", tool_call_id="call-b")
    c.record_persisted_result("call-b", "/tmp/spill-b.md")
    c.reset_for_turn()

    from agent.tool_guardrails import _session_repeat_message

    msg = _session_repeat_message(
        "read_file", 3, False, "same content",
        c._persisted_result_paths, first_call_id="call-a",
    )
    assert "/tmp/spill-a.md" in msg
    assert "/tmp/spill-b.md" not in msg


def test_session_repeat_message_omits_note_without_own_path():
    """No spillover for THIS call -> no path note (never a foreign file)."""
    from agent.tool_guardrails import _session_repeat_message

    msg = _session_repeat_message(
        "read_file", 3, False, "same content",
        {"call-other": "/tmp/spill-other.md"}, first_call_id="call-mine",
    )
    assert "/tmp/spill-other.md" not in msg
    assert "persisted" not in msg


def test_session_observe_block_cites_own_signature_path():
    """End-to-end: the block decision built by session_observe carries the
    first-observed call's own spillover path. Mirrors the production ordering:
    session_observe (with tool_call_id) runs at tool completion, the spillover
    record lands right after (tool_executor._record_persisted_path_for_stub),
    and the pointer is read only when the block message is BUILT — next turn."""
    c = ToolCallGuardrailController(
        __import__("agent.tool_guardrails", fromlist=["ToolCallGuardrailConfig"]).ToolCallGuardrailConfig(
            session_warn_after=2, session_block_after=3,
        )
    )
    c.session_observe("read_file", {"path": "/a"}, "big result", tool_call_id="first-call-id")
    c.record_persisted_result("first-call-id", "/tmp/spill-own.md")
    c.session_observe("web_search", {"query": "x"}, "other result", tool_call_id="other-call-id")
    c.record_persisted_result("other-call-id", "/tmp/spill-other.md")
    c.reset_for_turn()

    d = c.session_observe("read_file", {"path": "/a"}, "big result", tool_call_id="first-call-id")
    assert d.action == "warn"
    c.reset_for_turn()
    d = c.session_observe("read_file", {"path": "/a"}, "big result", tool_call_id="first-call-id")
    assert d.action == "block"
    assert "/tmp/spill-own.md" in d.message
    assert "/tmp/spill-other.md" not in d.message
