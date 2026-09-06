"""Regression tests: OOMPolicy= dropped from systemd-run --scope argv (2026-09-06 incident).

OOMPolicy= is a service-unit property; transient scopes reject it ("Unknown
assignment", rc=1) on every systemd this property was sent to, fail-closing the
restart-safe cron dispatch probe — every cron job died at dispatch with zero
scrapes and zero alerts (upstream #102357 / #102486). The property must be
absent from BOTH the probe argv and the real spawn argv, while the
MemoryAccounting/MemoryMax containment stays. Mirrors the upstream fix under
review; delete or reduce when it merges and the tree goes stock.
"""

import subprocess
from unittest.mock import patch

import pytest

import tools.process_registry as pr


@pytest.fixture(autouse=True)
def _reset_module_caches():
    saved_avail = pr._SYSTEMD_SCOPE_AVAILABLE
    saved_probed_at = pr._SYSTEMD_SCOPE_PROBED_AT
    yield
    pr._SYSTEMD_SCOPE_AVAILABLE = saved_avail
    pr._SYSTEMD_SCOPE_PROBED_AT = saved_probed_at


def test_scope_argv_has_no_oompolicy():
    """The scope argv must never carry OOMPolicy= — scopes reject it outright."""
    argv = pr._systemd_scope_argv("/usr/bin/systemd-run", "unit-x", "/bin/true")
    assert "OOMPolicy=kill" not in argv
    # Isolation containment that scopes DO accept must survive.
    assert "MemoryAccounting=yes" in argv
    assert any(a.startswith("MemoryMax=") for a in argv)


def test_probe_succeeds_without_oompolicy():
    """End-to-end: systemd-run that accepts everything except OOMPolicy must not
    poison the probe (the property is no longer sent at all)."""
    def fake_run(argv, **kwargs):
        if "OOMPolicy=kill" in argv:
            return subprocess.CompletedProcess(
                [], 1, stdout="", stderr=b"Unknown assignment: OOMPolicy=kill"
            )
        return subprocess.CompletedProcess([], 0, stdout="", stderr=b"")

    with patch.object(pr.subprocess, "run", side_effect=fake_run):
        pr._SYSTEMD_SCOPE_AVAILABLE = None  # force a fresh probe
        pr._SYSTEMD_SCOPE_PROBED_AT = 0.0
        assert pr._systemd_run_user_scope_available() is True


def test_probe_still_fails_when_scoped_run_fails_for_real():
    """Guard the opposite edge: a genuinely unusable user bus must still fail
    closed — dropping the property must not mask real failures."""
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess([], 1, stdout="", stderr=b"Failed to connect to bus")

    with patch.object(pr.subprocess, "run", side_effect=fake_run):
        pr._SYSTEMD_SCOPE_AVAILABLE = None
        pr._SYSTEMD_SCOPE_PROBED_AT = 0.0
        assert pr._systemd_run_user_scope_available() is False
