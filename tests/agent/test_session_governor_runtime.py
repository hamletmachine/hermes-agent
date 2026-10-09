"""Runtime wiring of the Session-scope Repeat Governor (R1).

Proves the two runtime seams the unit tests can't:
- ``run_agent._append_guardrail_observation`` feeds ``session_observe`` (warn tier
  appends guidance to the result).
- ``tool_executor._dispatch_authorized_once`` consults ``session_before_call`` —
  a parked block from a previous turn's observation blocks the NEXT identical
  dispatch with a guardrail-refusal synthetic result.
"""

import json

import pytest

from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController
from agent.tool_result_classification import is_guardrail_refusal


def _controller():
    return ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=False)
    )


def test_warn_tier_appends_session_guidance_to_result():
    """The runtime's observation path appends the session notice to the result."""
    from run_agent import AIAgent

    controller = _controller()
    result = "same content"
    # Simulate the runtime sequence: two turns, identical call each.
    def _agent():
        return type("A", (), {
            "_tool_guardrails": controller,
            "_stall_guards_enabled": lambda self: False,
        })()

    function_result = AIAgent._append_guardrail_observation(
        _agent(), "read_file", {"path": "/a"}, result, failed=False, tool_call_id="c1",
    )
    assert function_result == result  # first sighting: untouched
    controller.reset_for_turn()
    function_result = AIAgent._append_guardrail_observation(
        _agent(), "read_file", {"path": "/a"}, result, failed=False, tool_call_id="c2",
    )
    assert "hermes note" in function_result
    assert "session" in function_result


def test_parked_block_gates_next_dispatch(monkeypatch):
    """After a 3rd cross-turn repeat, the NEXT identical dispatch is blocked with
    a guardrail refusal carrying the required key."""
    from agent import tool_executor

    controller = _controller()
    agent = type("A", (), {
        "_tool_guardrails": controller,
        "_stall_guards_enabled": lambda self: False,
        "_turns_since_memory": 0,
        "_iters_since_skill": 0,
    })()
    # Bind the REAL block-result builder so the refusal marker contract is tested
    # end to end (toolguard_synthetic_result carries guardrail_refusal: True).
    from agent.tool_guardrails import toolguard_synthetic_result
    agent._guardrail_block_result = lambda d: toolguard_synthetic_result(d)

    # Three cross-turn identical observations park the block.
    for _ in range(3):
        controller.session_observe("read_file", {"path": "/a"}, "same content")
        controller.reset_for_turn()
    assert controller._session_block_pending

    # The next dispatch of the same call hits the gate. Drive _dispatch_authorized_once
    # with an execute stub that must NOT run.
    executed = []

    class Ref:
        name = "read_file"
        args = {"path": "/a"}
        task_id = "t1"
        call_id = "c9"
        trace = []

        def emit_post(self, agent, result, **outcome):
            emitted.append(result)

    emitted = []
    state = tool_executor._ManagedToolResult(
        result=None, args={"path": "/a"}, middleware_trace=[], blocked=False, dispatched=False)

    # Patch the hooks the dispatcher consults before guardrails.
    monkeypatch.setattr("agent.tool_executor._pre_tool_block", lambda agent, ref: (None, ref.args))

    result = tool_executor._dispatch_authorized_once(
        agent, state, Ref(),
        execute=lambda args: executed.append(args) or "SHOULD NOT RUN",
        scope_block=None, display_index=None,
        begin_execution=None, authorization_gate=None,
    )

    assert not executed, "the blocked call must never reach the tool"
    assert is_guardrail_refusal(result)
    assert "session" in result.lower()


def test_session_block_consumed_exactly_once():
    controller = _controller()
    for _ in range(3):
        controller.session_observe("read_file", {"path": "/a"}, "same content")
        controller.reset_for_turn()
    first = controller.session_before_call("read_file", {"path": "/a"})
    second = controller.session_before_call("read_file", {"path": "/a"})
    assert first is not None and first.action == "block"
    assert second is None  # consumed