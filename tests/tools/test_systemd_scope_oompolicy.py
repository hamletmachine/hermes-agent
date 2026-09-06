"""Regression tests: OOMPolicy= systemd-version gate (2026-09-06 incident).

systemd <254 rejects `OOMPolicy=kill` with "Unknown assignment" (rc=1). The
restart-safe cron dispatch probe used to send that property unconditionally,
so on Ubuntu 22.04 (systemd 249) every probe failed -> every cron job died at
dispatch ("systemd-run --user --scope is unavailable") with zero scrapes and
zero alerts. The gate must drop the property on old systemd and keep it on
>=254, and the end-to-end probe must succeed again on a manager that rejects
OOMPolicy but accepts the rest of the argv.
"""

import subprocess
from unittest.mock import patch

import pytest

import tools.process_registry as pr


@pytest.fixture(autouse=True)
def _reset_module_caches():
    saved_version = pr._SYSTEMD_VERSION
    saved_avail = pr._SYSTEMD_SCOPE_AVAILABLE
    saved_probed_at = pr._SYSTEMD_SCOPE_PROBED_AT
    yield
    pr._SYSTEMD_VERSION = saved_version
    pr._SYSTEMD_SCOPE_AVAILABLE = saved_avail
    pr._SYSTEMD_SCOPE_PROBED_AT = saved_probed_at


def test_oompolicy_dropped_on_systemd_249(monkeypatch):
    monkeypatch.setattr(pr, "_SYSTEMD_VERSION", 249)
    argv = pr._systemd_scope_argv("/usr/bin/systemd-run", "unit-x", "/bin/true")
    assert "OOMPolicy=kill" not in argv
    # The isolation properties that systemd 249 DOES support must survive.
    assert "MemoryAccounting=yes" in argv
    assert any(a.startswith("MemoryMax=") for a in argv)


def test_oompolicy_kept_on_systemd_254(monkeypatch):
    monkeypatch.setattr(pr, "_SYSTEMD_VERSION", 254)
    argv = pr._systemd_scope_argv("/usr/bin/systemd-run", "unit-y", "/bin/true")
    assert "OOMPolicy=kill" in argv


def test_version_detection_parses_systemctl_output(monkeypatch):
    fake = subprocess.CompletedProcess([], 0, stdout="systemd 249 (249.11-0ubuntu3.22)\n+PAM\n")
    with patch.object(pr.subprocess, "run", return_value=fake):
        pr._SYSTEMD_VERSION = None
        assert pr._systemd_supports_oom_policy() is False
        assert pr._SYSTEMD_VERSION == 249
    with patch.object(pr.subprocess, "run", return_value=fake):
        pr._SYSTEMD_VERSION = None
        # Undetectable version falls OPEN: probe failure then surfaces the real error.
        pr._SYSTEMD_VERSION = 0
        assert pr._systemd_supports_oom_policy() is False
    pr._SYSTEMD_VERSION = None


def test_probe_succeeds_on_manager_rejecting_oompolicy(monkeypatch):
    """End-to-end regression: a systemd-run that fails ONLY on OOMPolicy= must
    no longer poison the probe once the version gate drops the property."""
    monkeypatch.setattr(pr, "_SYSTEMD_VERSION", 249)

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


def test_probe_still_fails_when_scoped_run_fails_for_real(monkeypatch):
    """Guard the opposite edge: a genuinely unusable user bus must still fail
    closed — the version gate must not mask real failures."""
    monkeypatch.setattr(pr, "_SYSTEMD_VERSION", 254)

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess([], 1, stdout="", stderr=b"Failed to connect to bus")

    with patch.object(pr.subprocess, "run", side_effect=fake_run):
        pr._SYSTEMD_SCOPE_AVAILABLE = None
        pr._SYSTEMD_SCOPE_PROBED_AT = 0.0
        assert pr._systemd_run_user_scope_available() is False
