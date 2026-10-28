"""Schema-shape tests for the built-in memory tool.

The memory tool previously used ``allOf: [{if: ..., then: {required: ...}}]``
at the top level of ``parameters`` to hint per-action required fields.  That
form was:

  1. Ignored by every provider (Chat Completions doesn't honour ``if/then``
     on function schemas), so it never actually enforced anything.
  2. **Rejected outright by strict backends** — OpenAI's Codex endpoint
     (``chatgpt.com/backend-api/codex``, gpt-5.x) returns
     ``Invalid schema for function 'memory': schema must have type 'object'
     and not have 'oneOf'/'anyOf'/'allOf'/'enum'/'not' at the top level``.

We now rely on the runtime handler (``memory_tool()`` in ``tools/memory_tool.py``)
to validate required fields per action and return actionable error messages.
These tests guard the schema against regressing back to a shape strict
backends reject.
"""

from tools.memory_tool import MEMORY_SCHEMA

_FORBIDDEN_TOP_LEVEL_KEYS = ("allOf", "anyOf", "oneOf", "enum", "not")

def test_memory_schema_has_no_forbidden_top_level_combinators():
    """OpenAI's Codex backend rejects these at the top level of parameters."""
    params = MEMORY_SCHEMA["parameters"]
    for key in _FORBIDDEN_TOP_LEVEL_KEYS:
        assert key not in params, (
            f"top-level {key!r} in memory tool parameters will break the "
            "Codex backend (chatgpt.com/backend-api/codex). Per-action "
            "required-field checks belong in the runtime handler, not the schema."
        )


def test_memory_schema_requires_payload_fields():
    """The thinking DeepSeek variant drops optional arguments; require them.

    A call that arrives without old_text/content is rejected at runtime, but the
    model then tends to repeat the identical bare call. Schema-required fields
    force the provider to emit them. Empty strings are allowed for actions that
    do not use a field (add/remove); the runtime handler still validates real
    patch/remove payloads.
    """
    params = MEMORY_SCHEMA["parameters"]
    assert set(params["required"]) == {"target", "content", "old_text"}
    item = params["properties"]["operations"]["items"]
    assert set(item["required"]) == {"action", "content", "old_text"}
