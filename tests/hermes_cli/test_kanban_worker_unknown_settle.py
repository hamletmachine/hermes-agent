"""Forced contract: a kanban worker's claim receipt resolving ``unknown`` must be
SETTLED (resolve_unknown) and re-polled, not treated as a fatal crash (the epoch-rotation
wedge that produced 7 crash journals + alert fan-out on 2026-10-26).

Red on pristine base: the old client raises GatewayClientError on the first unknown
receipt and never calls resolve_unknown.
"""
import json, sqlite3, sys, time
from pathlib import Path

import pytest


class FakeClient:
    """Scripts the receipt sequence; records resolve_unknown calls."""

    def __init__(self, receipts=None, turn_unknown_into_terminal_after_resolve=True):
        self._receipts = list(receipts or [])
        self._calls = []
        self.turn_unknown_into_terminal_after_resolve = turn_unknown_into_terminal_after_resolve
        self.resolved = []

    async def rpc(self, method, **params):
        self._calls.append((method, params))
        if method == "kanban.run":
            sid = "local-testsided"
            first = self._receipts.pop(0)
            return {"session_id": sid, "receipt": dict(first, session_id=sid)}
        if method == "prompt.receipt":
            if self._receipts:
                return dict(self._receipts.pop(0))
            return dict(self._last)
        if method == "prompt.resolve_unknown":
            self.resolved.append(params)
            if self.turn_unknown_into_terminal_after_resolve:
                self._receipts.insert(0, {"status": "terminal", "outcome": "interrupted",
                                          "admission_id": self._last["admission_id"],
                                          "execution_generation": 3})
            return {"ok": True}
        raise AssertionError(method)

    def push_receipt(self, receipt):
        self._last = dict(receipt)
        self._receipts.append(dict(receipt))


def _board_db(tmp_path: Path, claim_lock: str = "lock-1", exit_code: int = 0, book_result: bool = True) -> str:
    db = tmp_path / "board.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE task_events (id INTEGER PRIMARY KEY, task_id TEXT, run_id INTEGER,"
                " kind TEXT, payload TEXT, created_at INTEGER)")
    if not book_result:
        con.commit(); con.close(); return str(db)
    con.execute("INSERT INTO task_events (task_id, run_id, kind, payload) VALUES (?,?,?,?)",
                ("t_x", 5, "worker_result",
                 json.dumps({"claim_lock": claim_lock, "exit_code": exit_code,
                             "summary": "done"})))
    con.commit()
    con.close()
    return str(db)


@pytest.mark.platforms("linux")
def test_unknown_claim_receipt_is_resolved_and_settled(tmp_path, monkeypatch):
    from hermes_cli.kanban_worker_client import _submit_claim
    board_db = _board_db(tmp_path)
    client = FakeClient([])
    client.push_receipt({"status": "queued", "admission_id": "adm-1", "execution_generation": 3})
    client.push_receipt({"status": "unknown", "admission_id": "adm-1", "execution_generation": 3})

    async def run():
        return await _submit_claim(client, {"board": "b", "task_id": "t_x", "run_id": 5,
                                            "claim_lock": "lock-1"}, board_db)
    import asyncio
    out = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(run())
    assert any(m == "prompt.resolve_unknown" for m, _ in client._calls), client._calls
    assert client.resolved[0]["admission_id"] == "adm-1"
    assert out == 0, out


@pytest.mark.platforms("linux")
def test_unknown_receipt_without_terminal_still_settles_and_reports_honestly(tmp_path, monkeypatch):
    """Resolution acked but the row never reaches terminal: the worker must exit through the
    honest worker_result path (no GatewayClientError crash journal), disposition recorded."""
    from hermes_cli import kanban_worker_client as kwc
    from hermes_cli.kanban_worker_client import _submit_claim
    monkeypatch.setattr(kwc, "UNKNOWN_RESOLVE_WAIT_S", 1)
    board_db = _board_db(tmp_path, book_result=False)  # the turn never ran: no booked result
    client = FakeClient(turn_unknown_into_terminal_after_resolve=False)
    client.push_receipt({"status": "queued", "admission_id": "adm-2", "execution_generation": 3})
    client.push_receipt({"status": "unknown", "admission_id": "adm-2", "execution_generation": 3})

    import asyncio
    out = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        _submit_claim(client, {"board": "b", "task_id": "t_x", "run_id": 5,
                               "claim_lock": "lock-1"}, board_db))
    assert any(m == "prompt.resolve_unknown" for m, _ in client._calls)
    assert out == 1, out  # honest terminal: exit_code 1 (no crash-raise)
