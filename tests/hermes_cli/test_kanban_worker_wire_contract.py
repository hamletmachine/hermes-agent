"""DISCRIMINATING wire contract: the resolve call must carry EXACTLY
{"session_id", "admission_id", "execution_generation"} — every other caller in the tree
sends ``execution_generation`` (sessions_discard, gateway_chat_view, ui-tui, desktop
canonical). ``generation=`` is refused 4001 invalid_params by the dispatcher's param gate
(gateway/session_controls.py:267-272) and again by the handler (:742-744), which recreates
the worker exit-1 crash through GatewayRPCError -> caught in _submit() -> nonzero_exit.
Red on 264726b833 (sends ``generation=``), green after the rename + guarded await.
"""
import asyncio, json, sqlite3
from pathlib import Path
import pytest


class WiredReplay:
    """Records the EXACT kwargs of every rpc call; replays scripted receipts for
    kanban.run / prompt.receipt; FAILS on resolve_unknown kwargs that violate the
    dispatcher's exact-key contract (the discrimination the old FakeClient lacked)."""

    ALLOWED_RESOLVE_KEYS = {"session_id", "admission_id", "execution_generation"}

    def __init__(self, receipts=None, terminal_after_resolve=True):
        self.receipts = list(receipts or [])
        self.calls = []
        self.resolves = []
        self.terminal_after_resolve = terminal_after_resolve
        self._adm = None

    async def rpc(self, method, **kw):
        self.calls.append((method, dict(kw)))
        if method == "kanban.run":
            first = self.receipts.pop(0)
            self._adm = first["admission_id"]
            return {"session_id": "local-wired", "receipt": dict(first, session_id="local-wired")}
        if method == "prompt.receipt":
            return dict(self.receipts.pop(0)) if self.receipts else {"status": "unknown",
                                                                     "admission_id": self._adm}
        if method == "prompt.resolve_unknown":
            extra = set(kw) - self.ALLOWED_RESOLVE_KEYS
            if extra:
                # the dispatcher's param gate behaviour verbatim: refusal BEFORE the handler
                from hermes_cli.gateway_client import GatewayClientError
                raise GatewayClientError(f"invalid_params: {sorted(extra)}")
            self.resolves.append(dict(kw))
            if self.terminal_after_resolve:
                self.receipts.insert(0, {"status": "terminal", "outcome": "interrupted",
                                         "admission_id": self._adm})
            return {"ok": True}
        raise AssertionError(method)


def _board_db(tmp_path, book_result=True):
    db = tmp_path / "board.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE task_events (id INTEGER PRIMARY KEY, task_id TEXT, run_id INTEGER,"
                " kind TEXT, payload TEXT, created_at INTEGER)")
    if book_result:
        con.execute("INSERT INTO task_events (task_id, run_id, kind, payload) VALUES (?,?,?,?)",
                    ("t_x", 5, "worker_result", json.dumps({"claim_lock": "lock-1", "exit_code": 0})))
    con.commit(); con.close()
    return str(db)


def _run(board_db, client):
    import hermes_cli.kanban_worker_client as kwc
    from hermes_cli.kanban_worker_client import _submit_claim
    old = kwc.UNKNOWN_RESOLVE_WAIT_S
    kwc.UNKNOWN_RESOLVE_WAIT_S = 1
    try:
        return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            _submit_claim(client, {"board": "b", "task_id": "t_x", "run_id": 5,
                                   "claim_lock": "lock-1"}, board_db))
    finally:
        kwc.UNKNOWN_RESOLVE_WAIT_S = old


@pytest.mark.platforms("linux")
def test_resolve_wire_carries_execution_generation_and_settles(tmp_path):
    board_db = _board_db(tmp_path)
    c = WiredReplay([{"status": "queued", "admission_id": "adm-1", "execution_generation": 3},
                     {"status": "unknown", "admission_id": "adm-1", "execution_generation": 3}])
    out = _run(board_db, c)
    assert c.resolves, "resolve_unknown was never called"
    assert c.resolves[0].keys() == {"session_id", "admission_id", "execution_generation"}, c.resolves[0]
    assert c.resolves[0]["execution_generation"] == 3
    assert out == 0, out


@pytest.mark.platforms("linux")
def test_resolve_refusal_is_nonfatal_honest_exit(tmp_path):
    """Unguarded await re-created exit-1 on ANY owner refusal (stale_generation /
    permission_denied). The worker must continue to the worker_result path."""
    board_db = _board_db(tmp_path, book_result=False)
    c = WiredReplay([{"status": "queued", "admission_id": "adm-2", "execution_generation": 7},
                     {"status": "unknown", "admission_id": "adm-2", "execution_generation": 7}],
                    terminal_after_resolve=False)

    async def refuse(method, **kw):
        from hermes_cli.gateway_client import GatewayClientError
        if method == "prompt.resolve_unknown":
            raise GatewayClientError("stale_generation")
        return await WiredReplay.rpc(c, method, **kw)
    c.rpc = refuse
    out = _run(board_db, c)
    assert out == 1, out  # honest ledger exit, no crash-raise
