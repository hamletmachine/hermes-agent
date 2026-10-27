"""The model-facing completion schema must require a handoff summary.

Before 2026-10-27 the schema said ``required: []`` while the handler rejected an
empty call. Workers emitted ``kanban_complete(board=...)``, got the handler
error, and looped into a protocol violation. The schema is the part the model
sees; requiring ``summary`` here prevents generating that malformed call.
"""

from tools.kanban_tools_schemas import KANBAN_COMPLETE_SCHEMA


def test_complete_requires_summary_in_model_schema():
    assert "summary" in KANBAN_COMPLETE_SCHEMA["parameters"]["required"]


def test_complete_still_exposes_legacy_result_for_cli_compat():
    assert "result" in KANBAN_COMPLETE_SCHEMA["parameters"]["properties"]
