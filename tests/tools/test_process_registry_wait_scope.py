"""R2-lite: process_registry.wait() must read TERMINAL_TIMEOUT scope-aware.

``wait()`` read ``os.getenv("TERMINAL_TIMEOUT")`` raw — the ONLY terminal setting
consumer still bypassing ``tools.terminal_scope``. Under the multiplexed gateway the
routed profile's ``TERMINAL_*`` policy (from its ``.env``) lives in the per-turn
terminal-scope ContextVar, deliberately NOT in ``os.environ`` (#68559 anti-leak
design) — so red's ``TERMINAL_TIMEOUT=60`` never reached the wait clamp and waits
genuinely ran 175s (red session 20261008_204837_a0d2da52). Tests here force that:
a profile policy of 60 must clamp a 175s request even with a clean process env.
"""

import subprocess
import time

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession
from tools.terminal_scope import set_terminal_scope, reset_terminal_scope


@pytest.fixture()
def registry():
    return ProcessRegistry()


@pytest.fixture()
def _no_ambient_timeout(monkeypatch):
    """No launch-env TERMINAL_TIMEOUT: the old os.getenv read falls to 180."""
    monkeypatch.delenv("TERMINAL_TIMEOUT", raising=False)


def _exited_session(registry, sid="proc_scope_wait"):
    """A session whose process exits quickly — wait() returns via the reconcile
    path with any clamp note attached (fast, no real waiting)."""
    proc = subprocess.Popen(["sh", "-c", "exit 0"])
    s = ProcessSession(
        id=sid, command="echo hello", task_id="t1",
        started_at=time.time(), output_buffer="",
    )
    s.process = proc
    s.pid = proc.pid
    registry._running[s.id] = s
    return s


def test_wait_clamps_to_profile_scope_timeout(registry, _no_ambient_timeout):
    """TERMINAL_TIMEOUT=60 from the profile policy clamps a 175s request (forced
    trigger: old code read os.getenv -> 180 -> no clamp note)."""
    _exited_session(registry)
    token = set_terminal_scope({"TERMINAL_TIMEOUT": "60"})
    try:
        result = registry.wait("proc_scope_wait", timeout=175)
    finally:
        reset_terminal_scope(token)
    assert result["status"] == "exited", result
    assert "clamped to configured limit of 60s" in (result.get("timeout_note") or ""), result


def test_wait_without_scope_keeps_process_env_default(registry, _no_ambient_timeout, monkeypatch):
    """No scope bound (standalone CLI): os.environ still wins — unchanged behavior."""
    _exited_session(registry)
    monkeypatch.setenv("TERMINAL_TIMEOUT", "60")
    result = registry.wait("proc_scope_wait", timeout=175)
    assert result["status"] == "exited", result
    assert "clamped to configured limit of 60s" in (result.get("timeout_note") or ""), result
