"""Forced-trigger test: a bare memory patch (missing required args) must be
rejected before dispatch with a retryable error — never executed.

Base behavior: valid-JSON-but-missing-required args reach the tool handler,
which returns success:false; the model then tends to repeat the identical bare
call or quit with a canned apology (observed 2026-10-28, red + purple).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from run_agent import AIAgent


def _agent(memory_required):
    agent = AIAgent.__new__(AIAgent)
    agent.provider, agent.model, agent.session_id = "nous", "m", "s1"
    agent.tools = [{
        "type": "function",
        "function": {
            "name": "memory",
            "parameters": {"type": "object", "required": memory_required},
        },
    }]
    agent.valid_tool_names = {"memory"}
    agent.log_prefix = ""
    agent._invalid_tool_retries = agent._invalid_json_retries = 0
    agent.api_mode = "chat_completions"
    agent.provider = "nous"
    agent._base_url = agent.base_url = "https://example.invalid/v1"
    agent.model = "m"
    agent.session_id = "s1"
    return agent


@pytest.fixture(autouse=True)
def _no_metrics(monkeypatch):
    import hermes_cli.observability.shared_metrics_model as metrics

    monkeypatch.setattr(metrics, "record_tool_call_quality", lambda *a, **k: None)


def _run(agent, arguments):
    from agent.turn_tool_validation import validate_tool_calls

    tc = SimpleNamespace(id="call_test_1", type="function",
                         function=SimpleNamespace(name="memory", arguments=arguments))
    message = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    verdict = validate_tool_calls(
        agent, message, "tool_calls", messages=messages,
        conversation_history=[], api_call_count=1, effective_task_id="t1",
    )
    return verdict, messages


def test_bare_memory_patch_is_refused_before_dispatch():
    """Valid JSON but missing a schema-required arg → retry, not dispatch.

    The invalid-JSON machinery retries the API call first (no message appended);
    once the retry budget is spent it injects one tool error result per call so
    the model can recover in-band. Drive both phases."""
    agent = _agent(["target", "content", "old_text"])
    for _ in range(2):
        verdict, messages = _run(agent, '{"target": "memory", "action": "patch"}')
        assert verdict.action == "continue", verdict
    assert [m for m in messages if m.get("role") == "tool"] == []
    # 3rd validation (retries exhausted) injects the recovery results.
    verdict, messages = _run(agent, '{"target": "memory", "action": "patch"}')
    assert verdict.action == "continue", verdict
    tool_results = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_results) == 1
    assert "old_text" in tool_results[0]["content"]


def test_complete_memory_patch_dispatches_normally():
    verdict, messages = _run(_agent(["target", "content", "old_text"]),
                             '{"target": "memory", "action": "patch", "old_text": "x", "content": "y"}')
    assert verdict.action == "ok"
    assert [m for m in messages if m.get("role") == "tool"] == []


def test_tool_without_required_fields_is_unaffected():
    verdict, _ = _run(_agent([]), '{"target": "memory"}')
    assert verdict.action == "ok"


def test_missing_arg_in_batch_op_is_caught():
    """A batch op omitting required fields is refused even when the top level is complete."""
    agent = _agent(["target"])
    agent.tools[0]["function"]["parameters"]["properties"] = {}
    agent.tools[0]["function"]["parameters"]["properties"]["operations"] = {
        "type": "array",
        "items": {"type": "object", "required": ["action", "content", "old_text"]},
    }
    args = json.dumps({"target": "memory",
                       "operations": [{"action": "patch"}],
                       "content": "", "old_text": ""})
    verdict, messages = _run(agent, args)
    assert verdict.action == "continue"
    assert [m for m in messages if m.get("role") == "tool"] == []
    verdict, messages = _run(agent, args)
    assert verdict.action == "continue"
    assert [m for m in messages if m.get("role") == "tool"] == []
    # Third attempt: retry budget spent → recovery results injected.
    verdict, messages = _run(agent, args)
    assert verdict.action == "continue"
    tool_results = [m for m in messages if m.get("role") == "tool"]
    assert tool_results and "operations[1]" in tool_results[0]["content"]  # ops are 1-indexed in the error
