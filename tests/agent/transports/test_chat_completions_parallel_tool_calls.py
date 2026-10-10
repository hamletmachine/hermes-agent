"""C1 — parallel_tool_calls on the chat-completions path (forced-trigger tests).

Review mandate:
- The flag rides the EXISTING seam: ``ProviderProfile.build_api_kwargs_extras``
  top-level kwargs — no bespoke config key.
- ``model.parallel_tool_calls: auto|true|false`` is an explicit user override
  that wins over the profile; ``auto`` (default/UNSET) omits the key entirely.
- ``true`` is capability-gated (strict endpoints 400/422 on unknown top-level
  fields — same failure class as reasoning_details).

These FAILED on the pre-change code: the key was never sent on this path.
"""

import pytest

from providers.base import ProviderProfile
from agent.transports.chat_completions import ChatCompletionsTransport


def _build(profile=None, model="test-model", tools=None, params=None):
    transport = ChatCompletionsTransport()
    messages = [{"role": "user", "content": "hi"}]
    kwargs = transport.build_kwargs(
        model=model, messages=messages, tools=tools or [], **(params or {}),
    ) if profile is None else transport.build_kwargs(
        model=model, messages=messages, tools=tools or [],
        provider_profile=profile, **(params or {}),
    )
    return kwargs


def _tool():
    return {"type": "function", "function": {
        "name": "search_files", "description": "search",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}},
    }}


def test_default_omits_parallel_tool_calls():
    kwargs = _build(tools=[_tool()])
    assert "parallel_tool_calls" not in kwargs


def test_profile_declaring_parallel_tool_calls_sends_it():
    profile = ProviderProfile(
        name="ptest", supports_parallel_tool_calls=True,
    )
    kwargs = _build(profile=profile, tools=[_tool()])
    assert kwargs.get("parallel_tool_calls") is True


def test_profile_capability_false_never_sends():
    profile = ProviderProfile(name="ptest", supports_parallel_tool_calls=False)
    kwargs = _build(profile=profile, tools=[_tool()])
    assert "parallel_tool_calls" not in kwargs


def test_model_override_true_wins_without_capability(monkeypatch):
    """``model.parallel_tool_calls: true`` is an explicit user override: it wins
    over the profile default and over the capability gate (user's explicit
    choice — the strict-endpoint 400 is theirs to hit)."""
    monkeypatch.setattr(
        "agent.transports.chat_completions._model_parallel_tool_calls_override",
        lambda model, provider=None: True, raising=False,
    )
    profile = ProviderProfile(name="ptest", supports_parallel_tool_calls=False)
    kwargs = _build(profile=profile, tools=[_tool()])
    assert kwargs.get("parallel_tool_calls") is True


def test_model_override_false_sends_explicit_false():
    """``false`` disables batching ON THE WIRE (explicit false beats the profile's
    capability-true); only ``auto``/UNSET omits the key."""
    profile = ProviderProfile(name="ptest", supports_parallel_tool_calls=True)
    kwargs = _build(
        profile=profile, tools=[_tool()],
        params={"model_parallel_tool_calls": False},
    )
    assert kwargs.get("parallel_tool_calls") is False


def test_model_override_auto_omits():
    profile = ProviderProfile(name="ptest", supports_parallel_tool_calls=True)
    kwargs = _build(
        profile=profile, tools=[_tool()],
        params={"model_parallel_tool_calls": "auto"},
    )
    assert "parallel_tool_calls" not in kwargs


def test_no_tools_no_flag():
    """codex path pops the flag when tools are absent — strict backends 400 on it
    without tools (chat_completion_helpers.py:2232). Same rule here."""
    profile = ProviderProfile(name="ptest", supports_parallel_tool_calls=True)
    kwargs = _build(profile=profile, tools=[])
    assert "parallel_tool_calls" not in kwargs
