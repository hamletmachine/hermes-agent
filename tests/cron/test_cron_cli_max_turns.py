"""C2 follow-up — CLI lane for the per-job max_turns cap.

The jobs/scheduler layer (create_job/update_job + resolution) shipped in
159c93f19a; these tests pin the CLI-only lane that reaches it (`hermes cron
create --max-turns` / `hermes cron edit --max-turns`). The MODEL lane is
intentionally ABSENT: max_turns is spend-adjacent sizing, exactly like model
pins — models must not point unattended spend at a different budget, so the
field is neither in CRONJOB_SCHEMA nor _HANDLER_FORWARDED_ARGS.
"""

import json
import pathlib
import tempfile
import types

import pytest

import cron.jobs as jobs
from cron.jobs import create_job, update_job
from hermes_cli import cron as cron_cli


@pytest.fixture()
def jobs_file(tmp_path):
    with jobs.use_cron_store(tmp_path):
        yield tmp_path / "cron" / "jobs.json"


def _fake_create_success(captured):
    def fake_api(**kwargs):
        captured.update(kwargs)
        return {"success": True, "job_id": "j1", "name": "n", "schedule": "0 9 * * *",
                "schedule_display": "daily 9am", "next_run_at": "2026-01-01T09:00:00Z",
                "skills": []}
    return fake_api


def test_create_forwards_max_turns(monkeypatch):
    captured = {}
    monkeypatch.setattr(cron_cli, "_cron_api", _fake_create_success(captured))
    rc = cron_cli.cron_command(types.SimpleNamespace(
        cron_command="create", schedule="0 9 * * *", prompt="do it", paused=False,
        max_turns=40))
    assert rc == 0
    assert captured["max_turns"] == 40


def test_create_omits_max_turns_when_unset(monkeypatch):
    captured = {}
    monkeypatch.setattr(cron_cli, "_cron_api", _fake_create_success(captured))
    rc = cron_cli.cron_command(types.SimpleNamespace(
        cron_command="create", schedule="0 9 * * *", prompt="do it", paused=False,
        max_turns=None))
    assert rc == 0
    # unset -> None -> create_job persists no key (byte-identical old jobs)
    assert captured["max_turns"] is None


def test_tool_lane_cannot_set_max_turns():
    """The model-facing surface must NOT expose max_turns (spend-adjacent sizing)."""
    from tools.cronjob_tools import CRONJOB_SCHEMA, _HANDLER_FORWARDED_ARGS
    assert "max_turns" not in CRONJOB_SCHEMA["parameters"]["properties"]
    assert "max_turns" not in _HANDLER_FORWARDED_ARGS


def test_cli_update_zero_clears_cap_via_tool_lane(jobs_file):
    """CLI edit --max-turns 0 drives cronjob(max_turns=0); the tool's update lane
    translates 0 -> None so update_job drops the stored key (clear = follow profile)."""
    from tools.cronjob_tools import cronjob as cronjob_tool

    job = create_job(prompt="x", schedule="0 9 * * *", name="clr", max_turns=40)
    result = json.loads(cronjob_tool(action="update", job_id=job["id"], max_turns=0))
    assert result["success"], result
    stored = json.loads(jobs_file.read_text())
    row = [j for j in stored["jobs"] if j["id"] == job["id"]][0]
    assert "max_turns" not in row  # cleared


def test_cli_update_positive_persists_via_tool_lane(jobs_file):
    from tools.cronjob_tools import cronjob as cronjob_tool

    job = create_job(prompt="x", schedule="0 9 * * *", name="cap")
    result = json.loads(cronjob_tool(action="update", job_id=job["id"], max_turns=25))
    assert result["success"], result
    stored = json.loads(jobs_file.read_text())
    row = [j for j in stored["jobs"] if j["id"] == job["id"]][0]
    assert row["max_turns"] == 25


def test_jobs_layer_rejects_zero():
    """0 stays invalid at the jobs layer — the CLI translates on its own lane only."""
    import pytest as _pytest

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "cron").mkdir()
        with jobs.use_cron_store(td):
            job = create_job(prompt="x", schedule="0 9 * * *")
            with _pytest.raises(ValueError):
                update_job(job["id"], {"max_turns": 0})
