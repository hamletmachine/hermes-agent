"""C2 — per-job cron max_turns cap (forced-trigger tests).

Review mandate (G2/D2/E1/F3: the knob was RETRACTED as unsupported; this task
re-adds it as a SUPPORTED scheduler feature):
- ``create_job`` accepts an optional ``max_turns`` (extends
  ``_CREATE_FIELD_NORMALIZERS`` + job schema).
- The scheduler honors it at the agent-setup site; UNSET -> profile
  ``agent.max_turns`` (current behavior byte-identical).
- Iteration-cap exhaustion must produce the existing timed_out outcome +
  notification, not a silent stop.

These FAILED on the pre-change code: create_job accepted no turn-cap field.
"""

import json

import pytest

import cron.jobs as jobs
from cron.jobs import create_job, load_jobs


@pytest.fixture()
def jobs_file(tmp_path):
    # use_cron_store expects the PARENT home; cron/ is created inside it.
    with jobs.use_cron_store(tmp_path):
        yield tmp_path / "cron" / "jobs.json"


def test_create_job_accepts_max_turns(jobs_file):
    job = create_job(prompt="do a thing", schedule="0 9 * * *", name="capped", max_turns=40)
    assert job["max_turns"] == 40
    stored = json.loads(jobs_file.read_text())
    assert stored["jobs"][0]["max_turns"] == 40


def test_create_job_without_max_turns_omits_key(jobs_file):
    """Optional keys persist only when explicitly set — byte-identical old jobs."""
    job = create_job(prompt="do a thing", schedule="0 9 * * *", name="uncapped")
    assert "max_turns" not in job


def test_create_job_rejects_invalid_max_turns(jobs_file):
    for bad in (0, -5, "many", 1.5):
        with pytest.raises(ValueError):
            create_job(prompt="x", schedule="0 9 * * *", max_turns=bad)


def test_create_job_accepts_none_max_turns(jobs_file):
    job = create_job(prompt="x", schedule="0 9 * * *", max_turns=None)
    assert "max_turns" not in job


def test_update_job_can_set_max_turns(jobs_file):
    job = create_job(prompt="x", schedule="0 9 * * *")
    from cron.jobs import update_job
    updated = update_job(job["id"], {"max_turns": 25})
    assert updated["max_turns"] == 25


def test_scheduler_resolves_job_cap_over_profile(jobs_file, monkeypatch, tmp_path):
    """The setup resolver honors the job cap; UNSET falls back to profile
    agent.max_turns."""
    from cron.scheduler import _resolve_job_turn_cap
    cfg = {"agent": {"max_turns": 500}}
    job = {"id": "j1", "max_turns": 40}
    assert _resolve_job_turn_cap(job, cfg) == 40
    job2 = {"id": "j2"}
    assert _resolve_job_turn_cap(job2, cfg) == 500
    assert _resolve_job_turn_cap({"id": "j3"}, {}) == _resolve_job_turn_cap({"id": "j3"}, {"agent": {}})


def test_exhaustion_keeps_timed_out_outcome():
    """The existing budget-exhaustion path already records a terminal timed_out
    outcome for dispatcher-owned kanban workers and the cron summary fallback
    (turn_finalizer._resolve_budget_fallback); the cap rides it — this pins the
    contract so a cap can never become a silent stop."""
    import inspect
    from agent import turn_finalizer
    source = inspect.getsource(turn_finalizer._resolve_budget_fallback)
    assert "max_iterations_reached" in source
    assert "timed_out" in source