"""Pending async completions are re-offered periodically and never dropped silently.

Discriminating fixtures for the t_3739406f contract (proposal before apply):

1. A pending row whose OWNER IS ALIVE is re-offered after ``_REOFFER_GAP_S`` (before this,
   ``sweep_orphaned_completions`` skipped alive-owner rows unconditionally, so a queue drop that
   was never popped wedged 'pending' forever — green profile's 14/14 never-delivered ledger).
2. Re-offers must not burn the attempt budget: eight failed claim cycles no longer converge to
   terminal 'dropped' within minutes (M7: the 8-attempt budget must not become an ~8-minute
   guillotine; a claim settlement adds one ``_REOFFER_GAP_S`` before the next offer, so the
   eight budget entries are real delivery pipelines, not one live sweep loop).
3. Every terminal ``dropped`` transition records a ``delivery_pointer`` line first: statuses,
   deliverable and transcript paths from the durable row's own ``result_json``, so the result
   stays findable from the ledger alone after the delivery attempt is gone.
4. An offer whose consumer never returns it is re-offered once its offer lease expires, instead
   of staying offered-forever while the row is pending.
"""

import json
import os
import queue
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tools import async_delegation as ad


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


class _Home:
    """Bind the profile home for code outside a turn, the way a delivery loop does."""

    def __init__(self, home: Path):
        self.home = home

    def __enter__(self):
        self._token = set_hermes_home_override(str(self.home))
        return self

    def __exit__(self, *exc):
        reset_hermes_home_override(self._token)


def _mk_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir(parents=True)
    return home


_RESULT = {
    "status": "completed",
    "summary": "wrote /home/island/ideas/report_and_findings.md and indexed /home/island/data/ledger.csv",
    "results": [
        {"task_index": 0, "status": "completed", "summary": "wrote /home/island/ideas/report_and_findings.md"},
        {"task_index": 1, "status": "error", "summary": "transcript /home/island/cache/task-1.log missing"},
    ],
    "live_transcripts": {"task-0": "/home/island/cache/delegation/live/x/task-0.log"},
}


def _persist_pending(home: Path, delegation_id: str, *, result: "dict | None" = None) -> float:
    """Create ONE terminal pending durable row the way the real finalize path does; returns updated_at."""
    evt = {
        "type": "async_delegation", "delegation_id": delegation_id,
        "session_key": "bot-chat", "status": "completed", "summary": "batch done",
        "dispatched_at": time.time() - 30, "completed_at": time.time() - 5,
    }
    base = {"status": "completed", "summary": "batch done"}
    with _Home(home):
        ad._persist_dispatch({
            "delegation_id": delegation_id,
            "session_key": "bot-chat",
            "origin_ui_session_id": "",
            "parent_session_id": "bot-parent",
            "dispatched_at": evt["dispatched_at"],
        })
        ad._persist_completion(evt, base if result is None else result)
        with sqlite3.connect(home / "state.db") as conn:
            conn.row_factory = sqlite3.Row
            row = dict(conn.execute(
                "SELECT * FROM async_delegations WHERE delegation_id=?", (delegation_id,)).fetchone())
    assert row["state"] == "completed" and row["delivery_state"] == "pending"
    assert row["owner_pid"] == os.getpid()  # dispatched from this process: the owner is alive
    return row["updated_at"]


def _set(home: Path, delegation_id: str, **cols) -> None:
    with sqlite3.connect(home / "state.db") as conn:
        conn.execute(f"UPDATE async_delegations SET {', '.join(f'{k}=?' for k in cols)} "
                     "WHERE delegation_id=?", (*cols.values(), delegation_id))


def _row(home: Path, delegation_id: str) -> dict:
    with sqlite3.connect(home / "state.db") as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute(
            "SELECT * FROM async_delegations WHERE delegation_id=?", (delegation_id,)).fetchone())


def _drain(q) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


# ── 1. Alive-owner staged re-offer ──────────────────────────────────────────

def test_pending_row_of_a_live_owner_is_re_offered_after_the_reoffer_gap(tmp_path):
    """The discard-shape incident: the queue drop was never claimed, so nothing re-offered it.
    The sweep now offers the alive-owner row once the re-offer gap passes, and a live offer
    (copy still unreturned) suppresses duplicates instead of flooding the queue."""
    home, q = _mk_home(tmp_path), queue.Queue()
    with _Home(home):
        updated = _persist_pending(home, "deleg_alive0001")
        # Fresh row: the in-process copy is still fresh, the sweep does not re-offer yet.
        assert ad.sweep_orphaned_completions(q, now=updated + 30) == 0
        assert ad.sweep_orphaned_completions(q, now=updated + 60) == 0
        assert q.empty()
        # First staged re-offer: the staleness gate responds like a dead-owner row would —
        # no restart and no cancelled copy ever, because the rescue no longer skips alive owners.
        assert ad.sweep_orphaned_completions(q, now=updated + 61) == 1
        (evt,) = _drain(q)
        assert evt["delegation_id"] == "deleg_alive0001"
        row = _row(home, "deleg_alive0001")
        assert row["delivery_state"] == "pending" and row["delivery_attempts"] == 0
        # The unreturned offer holds its lease: no duplicate queue copy within one gap.
        assert ad.sweep_orphaned_completions(q, now=updated + 121) == 0
        # Handed back (consumer discarded the copy with the row still pending): re-offerable.
        ad.return_completion_offer(evt)
        assert ad.sweep_orphaned_completions(q, now=updated + 181) == 1
        assert _row(home, "deleg_alive0001")["delivery_state"] == "pending"


def test_never_returned_offer_is_re_offered_once_the_lease_expires(tmp_path):
    """Offer lease expiry: a consumer that pops nothing (the 9d1ba058 shape — no claim was ever
    recorded) no longer wedges the row: each gap releases one fresh copy, never spending the
    attempt budget (attempts live at claim time, not at re-offer time)."""
    home, q = _mk_home(tmp_path), queue.Queue()
    with _Home(home):
        updated = _persist_pending(home, "deleg_alive0002")
        anchor = updated
        for _cycle in range(3):
            assert ad.sweep_orphaned_completions(q, now=anchor + ad._REOFFER_GAP_S) == 1
            _drain(q)  # nobody ever claims or returns these copies
            anchor += 2 * ad._REOFFER_GAP_S
        row = _row(home, "deleg_alive0002")
        assert row["delivery_state"] == "pending" and row["delivery_attempts"] == 0


# ── 2. Attempt-budget decoupling (M7) ───────────────────────────────────────

def test_failed_claim_cycles_no_longer_converge_within_minutes(tmp_path):
    """Eight failed claim cycles span at least seven re-offer gaps: the attempt budget stays
    the converging bound, but a delivery pipeline can no longer burn it in ~8 minutes."""
    home, q = _mk_home(tmp_path), queue.Queue()
    with _Home(home):
        t0 = _persist_pending(home, "deleg_alive0003")
        now = t0
        for cycle in range(1, 9):
            now = t0 + cycle * ad._REOFFER_GAP_S
            assert ad.sweep_orphaned_completions(q, now=now) == 1
            (_evt,) = _drain(q)
            claim_id = f"consumer:{cycle}"
            assert ad.claim_completion_delivery("deleg_alive0003", claim_id)
            assert ad.release_completion_delivery("deleg_alive0003", claim_id)
            if cycle < 8:
                assert _row(home, "deleg_alive0003")["delivery_state"] == "pending"
        row = _row(home, "deleg_alive0003")
        assert row["delivery_state"] == "dropped"
        assert json.loads(row["delivery_pointer"])["reason"] == "attempt_budget_release"
        assert (now - t0) >= 8 * ad._REOFFER_GAP_S - 1  # ~80 simulated minutes, not ~8


# ── 3. Pointer line on every terminal 'dropped' transition ─────────────────

def test_replay_age_drop_records_the_pointer(tmp_path):
    """Past the 48h replay cap the row converges to dropped, but the pointer line is written
    first: statuses and path-like deliverables from result_json survive the silent drop."""
    home, q = _mk_home(tmp_path), queue.Queue()
    with _Home(home):
        updated = _persist_pending(home, "deleg_ptr_age_01", result=_RESULT)
        dropped_at = updated + ad._MAX_COMPLETION_REPLAY_AGE_S + 1
        assert ad.sweep_orphaned_completions(q, now=dropped_at) == 0
        assert q.empty()
        row = _row(home, "deleg_ptr_age_01")
        assert row["delivery_state"] == "dropped"
        pointer = json.loads(row["delivery_pointer"])
        assert pointer["reason"] == "replay_age_cap"
        assert pointer["delegation_id"] == "deleg_ptr_age_01"
        statuses = [t["status"] for t in pointer["tasks"]]
        assert statuses == ["completed", "error"]
        paths = " ".join(pointer["deliverables"] + pointer["live_transcripts"])
        assert "/home/island/ideas/report_and_findings.md" in paths
        assert "/home/island/cache/delegation/live/x/task-0.log" in paths


def test_delivery_budget_drop_records_the_pointer(tmp_path):
    """The sweep- and release-side attempt-budget convergences both carry a pointer line."""
    home, q = _mk_home(tmp_path), queue.Queue()
    with _Home(home):
        # Sweep-side: a claimed-out row settles at the next sweep.
        updated = _persist_pending(home, "deleg_ptr_sw_01", result=_RESULT)
        _set(home, "deleg_ptr_sw_01", delivery_attempts=ad._MAX_DELIVERY_ATTEMPTS)
        assert ad.sweep_orphaned_completions(q, now=updated + ad._ORPHAN_STALE_S + 1) == 0
        row = _row(home, "deleg_ptr_sw_01")
        assert row["delivery_state"] == "dropped"
        assert json.loads(row["delivery_pointer"])["reason"] == "attempt_budget_sweep"
        # Release-side: the claimant's own capped release settles with a pointer.
        _persist_pending(home, "deleg_ptr_rl_01", result=_RESULT)
        assert ad.claim_completion_delivery("deleg_ptr_rl_01", "c1")
        # Raise the attempt count the way seven prior claim cycles would have.
        _set(home, "deleg_ptr_rl_01", delivery_attempts=ad._MAX_DELIVERY_ATTEMPTS)
        assert ad.release_completion_delivery("deleg_ptr_rl_01", "c1")
        row = _row(home, "deleg_ptr_rl_01")
        assert row["delivery_state"] == "dropped"
        assert json.loads(row["delivery_pointer"])["reason"] == "attempt_budget_release"


def test_deliberate_session_drop_records_the_pointer(tmp_path):
    """A preflight 'terminal' target (user boundary) drops the row deliberately: the pointer
    line still records what was dropped, from the durable row."""
    home, q = _mk_home(tmp_path), queue.Queue()
    with _Home(home):
        updated = _persist_pending(home, "deleg_ptr_dp_01", result=_RESULT)
        assert ad.sweep_orphaned_completions(q, now=updated + ad._REOFFER_GAP_S) == 1
        (evt,) = _drain(q)
        claim = ad.claim_event_delivery(evt, "gateway-test")
        assert claim
        assert ad.drop_completion_delivery("deleg_ptr_dp_01", claim)
        row = _row(home, "deleg_ptr_dp_01")
        assert row["delivery_state"] == "dropped"
        assert json.loads(row["delivery_pointer"])["reason"] == "session_terminal"


# ── 4. Released offer is sweep-eligible ────────────────────────────────────

def test_released_offer_is_sweepable_again_without_a_restart(tmp_path):
    """A consumer that claims, then discards its copy with the row still pending (a failed
    delivery) hands the offer back; the next sweep re-offers it. Pins the release seam that
    the gateway's failed-injection settlement uses."""
    home, q = _mk_home(tmp_path), queue.Queue()
    with _Home(home):
        updated = _persist_pending(home, "deleg_gw_hb_001")
        assert ad.sweep_orphaned_completions(q, now=updated + ad._REOFFER_GAP_S) == 1
        (evt,) = _drain(q)
        claim = ad.claim_event_delivery(evt, "gateway-test")
        assert claim
        attempts = _row(home, "deleg_gw_hb_001")["delivery_attempts"]
        assert attempts == 1
        ad.release_event_delivery(evt, claim)  # the gateway/TUI failed-injection settle seam
        row = _row(home, "deleg_gw_hb_001")
        assert row["delivery_state"] == "pending" and row["delivery_claim"] is None
        # The held offer was handed back: the sweep is eligible again without a restart.
        offer_key = (ad.hermes_home_key(str(home)), "deleg_gw_hb_001")
        assert offer_key not in ad._offered
        # One attempt settled, so the next offer waits the re-offer gap (M7 decoupling).
        assert ad.sweep_orphaned_completions(q, now=row["updated_at"] + ad._ORPHAN_STALE_S + 1) == 0
        assert ad.sweep_orphaned_completions(q, now=row["updated_at"] + ad._REOFFER_GAP_S + 1) == 1
        assert _row(home, "deleg_gw_hb_001")["delivery_state"] == "pending"
