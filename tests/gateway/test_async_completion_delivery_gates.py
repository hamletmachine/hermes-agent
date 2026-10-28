"""Async-completion delivery gates — unit tests for the 10-28 delivery fix.

Both functions under test are exec'd from the REAL sources in
gateway/run_notifications.py against stub collaborators (the reviewer's approach for
t_cee58630 round 1). Importing the full mixin would drag the entire gateway surface into
the harness, and `_asdi_primary_routes_for` is stubbed so a worktree test run never
touches the real Hermes home (home_io_guard refuses real-home reads).

Pinned behaviors (t_cee58630 BLOCK round + 10-28 satellite delivery defect):
  1. the raw api_server lane is NOT a delivery lane: gate logs OFF exactly once and
     returns False — t_cee58630 BLOCKER 1 (the instrumentation commit flipped this
     branch truthy by dropping one `return`);
  2. the shared-bot fallback resolves the route-authorized primary adapter for an
     authorized satellite chat (black: !VUeykrLKaYxWdAMluT:matrix.org);
  3. it fails CLOSED for an unauthorized chat and unserved profiles (never the wrong
     room);
  4. the primary profile keeps resolving through its own map; the satellite fallback
     guard only returns None when that resolution already failed.
"""
import asyncio
import ast
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "gateway" / "run_notifications.py"

BLACK_CHAT = "!VUeykrLKaYxWdAMluT:matrix.org"
DEFAULT_CHAT = "!jDTPjBCSBekKIWvCii:matrix.org"
_PRIMARY = "default"


class _StubAdapter:
    def __init__(self, name, supports_async_delivery=True):
        self.name = name
        self.supports_async_delivery = supports_async_delivery


class _Logger:
    def __init__(self):
        self.records = []

    def warning(self, msg, *args):
        self.records.append(("warning", msg % args if args else msg))

    def debug(self, msg, *args):
        self.records.append(("debug", msg % args if args else msg))


class _StubSource:
    def __init__(self, platform, chat_id, profile):
        self.thread_id = None
        self.platform = platform
        self.chat_id = chat_id
        self.profile = profile


class _StubRoute:
    """Attribute-shaped stand-in for gateway.profile_routing.ProfileRoute."""

    def __init__(self, profile, chat_id=BLACK_CHAT):
        self.name = ""
        self.platform = "matrix"
        self.profile = profile
        self.guild_id = None
        self.chat_id = chat_id
        self.thread_id = None
        self.enabled = True
        self.bot_profile = None
        self.user_id = None

    def matches(self, platform, guild_id=None, chat_id=None, thread_id=None):
        return bool(self.chat_id) and chat_id == self.chat_id


def _module_namespace():
    import logging as _logging
    import gateway.run_notifications as real
    mod = types.ModuleType("under_test")
    mod.logging = _logging
    mod.asyncio = asyncio
    mod.logger = _Logger()
    mod.Platform = real.Platform
    mod._raw_process_event_session_id = real._raw_process_event_session_id
    mod._asdi_primary_routes_for = real._asdi_primary_routes_for
    return mod, mod.logger


def _exec(mod, fn_name, extra_globals=None):
    """Compile+exec the REAL function source, then read the bound name back.

    A `def` statement's expression value is None (the def binds the name), so eval alone
    returns None — read `glob[fn_name]` instead (bug found while wiring these tests).
    """
    src = SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == fn_name:
            segment = ast.get_source_segment(src, node)
            glob = dict(mod.__dict__, **(extra_globals or {}))
            exec(compile(segment, "<under_test>", "exec"), glob)
            return glob[fn_name]
    raise LookupError(fn_name)


def _resolver(monkeypatch, routes):
    """Exec the REAL `_resolve_injection_adapter`; stub the routes loader at the
    `_asdi_primary_routes_for` seam (module-name patched so the internal gateway.run
    import resolves to our stub — no real-home I/O)."""
    mod, log = _module_namespace().__dict__["logger"] and _module_namespace(), None
    mod = _module_namespace()
    log = mod.logger
    fake_run = types.ModuleType("gateway.run")
    fake_run._primary_routes_for_profile = lambda name: list(routes)
    monkeypatch.setitem(sys.modules, "gateway.run", fake_run)
    fn = _exec(mod, "_resolve_injection_adapter")
    return fn, mod, log


def _runner(own_map, primary_map):
    from gateway.run_notifications import Platform as _Plat

    def _enum_keys(mapping):
        return {_Plat(k) if isinstance(k, str) else k: v for k, v in (mapping or {}).items()}

    class Runner:
        _primary_profile_name = _PRIMARY

        def __init__(self):
            self._own = {name: _enum_keys(m) for name, m in (own_map or {}).items()}
            self._primary = _enum_keys(primary_map)
            self.config = type("Cfg", (), {"platforms": {}})()
            self.adapters = {}

        def _profile_adapters_map(self):
            return {name: dict(adapters) for name, adapters in self._own.items()} if self._own else {}

        def _adapters_for_profile(self, profile):
            if not profile or profile == self._primary_profile_name:
                return dict(self._primary)
            return dict(self._own.get(profile, {}))

        def _primary_adapters(self):
            return dict(self._primary)

        def _transport_owner(self, source):
            return None

    return Runner()


def _sync_to_thread(monkeypatch):
    monkeypatch.setattr(asyncio, "to_thread", lambda coro_fn, *a, **k: coro_fn(*a, **k), raising=False)


# ------------------------------------------------------------- raw api_server lane
def _ready_for_raw_lane(platform_cls, push_capable):
    self_stub = type("Self", (), {})()
    self_stub._build_process_event_source = staticmethod(lambda evt: None)
    self_stub.adapters = {platform_cls("API_SERVER"): _StubAdapter("api", push_capable)}
    mod, log = _module_namespace()
    wake_mod = __import__("gateway.wake", fromlist=["adapter_supports_push"])
    with mock.patch.object(
        wake_mod, "adapter_supports_push",
        lambda adapter: bool(getattr(adapter, "supports_async_delivery", False)), create=True,
    ):
        with mock.patch.dict(sys.modules, {"gateway.wake": wake_mod}, create=True):
            fn = _exec(mod, "_completion_delivery_ready")
            result = asyncio.run(fn(self_stub, {"session_key": "raw-session-abc"}))
    return result, log.records


@pytest.mark.parametrize("push_capable, expected_reason", [
    (True, "raw api_server session with push support"),
    (False, "adapter lacks push support and has no session db"),
])
def test_raw_api_server_lane_is_not_a_delivery_lane(push_capable, expected_reason):
    """t_cee58630 BLOCKER 1 regression guard: the gate logs OFF exactly once and returns
    False — it must never flip truthy on an adapter it itself declares undeliverable."""
    result, records = _ready_for_raw_lane(
        __import__("gateway.run_notifications", fromlist=["Platform"]).Platform, push_capable)
    assert result is False
    warnings = [m for level, m in records if level == "warning" and "delivery gate OFF" in m]
    assert len(warnings) == 1, f"exactly one gate-OFF line per event, got {warnings}"
    assert expected_reason in warnings[0]


def test_ready_reports_open_for_a_healthy_legacy_event(monkeypatch):
    """Source builds + adapter resolves + push supported → gate OPEN, and the OPEN line is
    logged at DEBUG (normal traffic must not spam WARNING cadence). Uses the REAL async
    asyncio.to_thread — the bound `_build_process_event_source` is a plain callable."""
    from gateway.run_notifications import Platform as _real_platform_cls
    mod, log = _module_namespace()
    wake_mod = __import__("gateway.wake", fromlist=["adapter_supports_push"])

    shared = _StubAdapter("shared-matrix")

    def make_source(evt):
        return _StubSource(_real_platform_cls("matrix"), BLACK_CHAT, "black")

    with mock.patch.object(wake_mod, "adapter_supports_push", lambda a: True, create=True), \
         mock.patch.dict(sys.modules, {"gateway.wake": wake_mod}, create=True):
        # Exec the fn with the stubs bound in ITS namespace (the function reads its own
        # globals; patching the real module doesn't reach it).
        import gateway.run_notifications as real
        ns = dict(mod.__dict__)
        ns["_build_process_event_source"] = make_source
        runner = _runner({"black": {"api_server": _StubAdapter("black-console")}}, {"matrix": shared})
        ns["_resolve_injection_adapter"] = lambda platform, source: shared
        runner._build_process_event_source = make_source
        runner._resolve_injection_adapter = lambda platform, source: shared
        fn = _exec(mod, "_completion_delivery_ready", extra_globals={
            "_build_process_event_source": make_source,
            "_resolve_injection_adapter": lambda platform, source: shared,
            "Platform": real.Platform,
        })
        # parent_session_id omitted: classification walks the sessions DB only when a parent
        # is present; the raw OPEN path is what this test pins.
        result = asyncio.run(fn(runner, {"session_key": "agent:black:matrix:dm:" + BLACK_CHAT}))
    assert result is True
    opens = [m for level, m in log.records if level == "debug" and "gate OPEN" in m]
    assert opens, "gate OPEN must be logged at debug"


# ----------------------------------------------------------- shared-bot fallback legs
def _resolver(monkeypatch, routes):
    mod, log = _module_namespace()
    fake_run = types.ModuleType("gateway.run")
    fake_run._primary_routes_for_profile = lambda name: list(routes)
    monkeypatch.setitem(sys.modules, "gateway.run", fake_run)
    fn = _exec(mod, "_resolve_injection_adapter")
    return fn, mod, log


def test_fallback_arms_the_route_authorized_shared_adapter(monkeypatch):
    """LEG A: black's own map lacks matrix; the profile_routes entry authorizes the black
    DM chat on the shared bot — the resolver returns the PRIMARY's matrix adapter."""
    _sync_to_thread(monkeypatch)
    fn, mod, log = _resolver(monkeypatch, [_StubRoute("black")])
    runner = _runner({"black": {"api_server": _StubAdapter("black-console")}}, {"matrix": "SHARED-MATRIX"})
    source = _StubSource(mod.Platform("matrix"), BLACK_CHAT, "black")
    assert fn(runner, "matrix", source) == "SHARED-MATRIX"
    warns = [m for level, m in log.records if level == "warning" and "injection fallback" in m]
    assert any("route-authorized shared adapter" in w for w in warns)


def test_fallback_fails_closed_on_wrong_chat(monkeypatch):
    """LEG B: a chat the route does not authorize stays unresolved (never the wrong room)."""
    _sync_to_thread(monkeypatch)
    fn, mod, log = _resolver(monkeypatch, [_StubRoute("black")])
    runner = _runner({"black": {"api_server": _StubAdapter("black-console")}}, {"matrix": "SHARED-MATRIX"})
    source = _StubSource(mod.Platform("matrix"), "!WRONGROOM:matrix.org", "black")
    assert fn(runner, "matrix", source) is None
    warns = [m for level, m in log.records if level == "warning" and "no content match" in m]
    assert warns, "fail-closed must be observable"


def test_fallback_refuses_unserved_profile(monkeypatch):
    """No route for the profile ⇒ fail closed (None) even if the primary owns the platform."""
    _sync_to_thread(monkeypatch)
    fn, mod, _log = _resolver(monkeypatch, [])
    runner = _runner({}, {"matrix": "SHARED-MATRIX"})
    source = _StubSource(mod.Platform("matrix"), BLACK_CHAT, "black")
    assert fn(runner, "matrix", source) is None


def test_primary_profile_resolves_via_own_map(monkeypatch):
    """The primary profile resolves through its own map; the satellite fallback guard is
    only reached when that resolution fails (returns None in that corner, never the
    shared-bot adapter for a wrong profile)."""
    _sync_to_thread(monkeypatch)
    fn, mod, _log = _resolver(monkeypatch, [_StubRoute(_PRIMARY, DEFAULT_CHAT)])
    runner = _runner({}, {"matrix": "PRIMARY-MATRIX"})
    source = _StubSource(mod.Platform("matrix"), DEFAULT_CHAT, _PRIMARY)
    assert fn(runner, "matrix", source) == "PRIMARY-MATRIX"


def test_primary_profile_with_broken_map_fails_closed(monkeypatch):
    """Corner: primary profile name passed but the own-map resolution came up empty —
    the guard returns None (fail closed) rather than silently arming the satellite path."""
    _sync_to_thread(monkeypatch)
    fn, mod, _log = _resolver(monkeypatch, [_StubRoute(_PRIMARY, DEFAULT_CHAT)])
    runner = _runner({}, {})  # primary map empty: pre-resolution fails for the primary too
    source = _StubSource(mod.Platform("matrix"), DEFAULT_CHAT, _PRIMARY)
    assert fn(runner, "matrix", source) is None
