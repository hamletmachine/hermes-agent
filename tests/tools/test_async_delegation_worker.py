"""Closed ledger authority and token-guarded retry contracts."""
import json
import os
import time

import pytest
import psutil

from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError
from tools import async_delegation as ledger


def test_worker_dispatch_is_bound_to_execution_and_cannot_replace_foreign_rows(tmp_path):
    from gateway.session_worker_delegation import worker_delegation_handlers
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('parent', source='cli')
    db.create_session('other', source='cli')
    handlers = worker_delegation_handlers('worker-a', os.getpid(), psutil.Process().create_time())
    payload = {'delegation_id': 'deleg_owned', 'task': {'goal': 'inert'}, 'dispatched_at': time.time(),
               'parent_session_id': 'parent', 'session_key': '', 'origin_ui_session_id': '', 'origin_session_id': ''}
    def apply(name, payload, sid='parent', table=handlers):
        return db._execute_write(lambda conn: table[name](db, conn, sid, payload))
    try:
        from contextlib import closing
        import sqlite3
        with closing(sqlite3.connect(tmp_path / 'state.db')) as conn:
            ledger._initialize_schema(conn)
        apply('delegation.dispatch', payload)
        foreign = worker_delegation_handlers('worker-b', os.getpid(), psutil.Process().create_time())
        for sid, table in [('other', handlers), ('parent', foreign)]:
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                apply('delegation.read', {'delegation_id': 'deleg_owned'}, sid, table)
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                apply('delegation.dispatch', dict(payload, parent_session_id=sid), sid, table)
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            apply('delegation.dispatch', dict(payload, delegation_id='bad', session_key='foreign-route'))
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            apply('delegation.read', {'delegation_id': 'deleg_owned', 'sql': 'DELETE FROM sessions'})
        with db._read_ctx() as conn:
            assert conn.execute('SELECT COUNT(*) FROM async_delegations').fetchone()[0] == 1
    finally:
        db.close()


def test_worker_completion_claims_refund_and_preserve_terminal_result(tmp_path):
    from gateway.session_worker_delegation import worker_delegation_handlers
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('parent', source='cli')
    handlers = worker_delegation_handlers('worker-a', os.getpid(), psutil.Process().create_time())
    def apply(name, **payload):
        return db._execute_write(lambda conn: handlers['delegation.' + name](db, conn, 'parent', payload))
    try:
        from contextlib import closing
        import sqlite3
        with closing(sqlite3.connect(tmp_path / 'state.db')) as conn:
            ledger._initialize_schema(conn)
        apply('dispatch', delegation_id='deleg_owned', task={'goal': 'inert'}, dispatched_at=time.time(),
              parent_session_id='parent', session_key='', origin_ui_session_id='', origin_session_id='')
        event = {'type': 'async_delegation', 'delegation_id': 'deleg_owned', 'parent_session_id': 'parent',
                 'session_key': '', 'origin_ui_session_id': '', 'origin_session_id': '', 'status': 'completed',
                 'completed_at': time.time(), 'summary': 'durable'}
        apply('complete', event=event, result={'summary': 'durable'})
        for index in range(10):
            token = f'claim-{index}'
            assert apply('claim', delegation_id='deleg_owned', claim_id=token)['value']
            assert not apply('ack', delegation_id='deleg_owned', claim_id='foreign')['value']
            assert apply('defer', delegation_id='deleg_owned', claim_id=token)['value']
        row = apply('read', delegation_id='deleg_owned')['value']
        assert row['delivery_attempts'] == 0 and row['delivery_state'] == 'pending'
        assert apply('claim', delegation_id='deleg_owned', claim_id='accepted')['value']
        assert apply('ack', delegation_id='deleg_owned', claim_id='accepted')['value']
        assert not apply('claim', delegation_id='deleg_owned', claim_id='late')['value']
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            apply('complete', event=dict(event, summary='overwrite'), result={'summary': 'overwrite'})
        with db._read_ctx() as conn:
            row = conn.execute('SELECT result_json,delivery_state FROM async_delegations').fetchone()
            assert json.loads(row[0]) == {'summary': 'durable'} and row[1] == 'delivered'
    finally:
        db.close()


_RESULT = {
    'status': 'completed',
    'summary': 'wrote /home/island/ideas/report_and_findings.md',
    'results': [
        {'task_index': 0, 'status': 'completed',
         'summary': 'wrote /home/island/ideas/report_and_findings.md'},
        {'task_index': 1, 'status': 'error',
         'summary': 'transcript /home/island/cache/task-1.log missing'},
    ],
    'live_transcripts': {'task-0': '/home/island/cache/delegation/live/x/task-0.log'},
}


def test_worker_terminal_drops_record_the_delivery_pointer(tmp_path):
    """The worker-ledger legs of drop and capped release converge through the local ledger's
    recorded-drop primitive: the delivery_pointer line rides the same UPDATE as the terminal
    'dropped' state, with the local leg's own reason tokens (t_5908f939 — these two RPC paths
    used to drop with no pointer at all). A release that is not past the attempt budget stays
    pending with its pointer column untouched, a payload-less row still converges (with a NULL
    pointer — nothing durable to cite), and a stale claim must not converge anything."""
    from gateway.session_worker_delegation import worker_delegation_handlers
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('parent', source='cli')
    handlers = worker_delegation_handlers('worker-a', os.getpid(), psutil.Process().create_time())
    cap = ledger._MAX_DELIVERY_ATTEMPTS

    def apply(name, **payload):
        return db._execute_write(lambda conn: handlers['delegation.' + name](db, conn, 'parent', payload))

    def seed(delegation_id):
        apply('dispatch', delegation_id=delegation_id, task={'goal': 'inert'},
              dispatched_at=time.time(), parent_session_id='parent', session_key='',
              origin_ui_session_id='', origin_session_id='')
        apply('complete', event={'type': 'async_delegation', 'delegation_id': delegation_id,
                                 'parent_session_id': 'parent', 'session_key': '',
                                 'origin_ui_session_id': '', 'origin_session_id': '',
                                 'status': 'completed', 'completed_at': time.time()},
              result=_RESULT)

    def state_of(delegation_id):
        with db._read_ctx() as conn:
            return dict(conn.execute(
                'SELECT delivery_state, delivery_pointer, delivery_claim FROM async_delegations '
                'WHERE delegation_id=?', (delegation_id,)).fetchone())

    def bump(delegation_id, attempts, *, strip_payload=False):
        def patch_row(conn):
            if strip_payload:
                conn.execute('UPDATE async_delegations SET event_json=NULL, result_json=NULL '
                             'WHERE delegation_id=?', (delegation_id,))
            conn.execute('UPDATE async_delegations SET delivery_attempts=? WHERE delegation_id=?',
                         (attempts, delegation_id))
        db._execute_write(patch_row)

    try:
        from contextlib import closing
        import sqlite3
        with closing(sqlite3.connect(tmp_path / 'state.db')) as conn:
            ledger._initialize_schema(conn)

        # drop with a durable payload: the pointer line rides the terminal UPDATE itself.
        seed('deleg_wdrop')
        assert apply('claim', delegation_id='deleg_wdrop', claim_id='c1')['value']
        assert apply('drop', delegation_id='deleg_wdrop', claim_id='c1')['value']
        row = state_of('deleg_wdrop')
        assert row['delivery_state'] == 'dropped' and row['delivery_claim'] is None
        pointer = json.loads(row['delivery_pointer'])
        assert pointer['reason'] == 'session_terminal' and pointer['delegation_id'] == 'deleg_wdrop'
        assert pointer['delivery_attempts'] == 1
        paths = ' '.join(pointer['deliverables'] + pointer['live_transcripts'])
        assert '/home/island/ideas/report_and_findings.md' in paths
        assert '/home/island/cache/delegation/live/x/task-0.log' in paths

        # drop with no stored payload: the row still converges; the pointer column holds NULL
        # (nothing durable to cite — no all-empty JSON placeholder).
        seed('deleg_wbare')
        assert apply('claim', delegation_id='deleg_wbare', claim_id='c2')['value']
        bump('deleg_wbare', 1, strip_payload=True)
        assert apply('drop', delegation_id='deleg_wbare', claim_id='c2')['value']
        row = state_of('deleg_wbare')
        assert row['delivery_state'] == 'dropped' and row['delivery_pointer'] is None

        # capped release carries its own pointer line with the local leg's budget token.
        seed('deleg_wcap')
        assert apply('claim', delegation_id='deleg_wcap', claim_id='c3')['value']
        bump('deleg_wcap', cap)
        assert apply('release', delegation_id='deleg_wcap', claim_id='c3')['value']
        row = state_of('deleg_wcap')
        assert row['delivery_state'] == 'dropped'
        assert json.loads(row['delivery_pointer'])['reason'] == 'attempt_budget_release'

        # release below the attempt budget: rent-free pending, pointer column untouched.
        seed('deleg_wopen')
        assert apply('claim', delegation_id='deleg_wopen', claim_id='c4')['value']
        assert apply('release', delegation_id='deleg_wopen', claim_id='c4')['value']
        row = state_of('deleg_wopen')
        assert row['delivery_state'] == 'pending'
        assert row['delivery_pointer'] is None and row['delivery_claim'] is None
        # A stale claim settles nothing, in either terminal direction.
        assert not apply('drop', delegation_id='deleg_wopen', claim_id='foreign')['value']
        assert not apply('release', delegation_id='deleg_wopen', claim_id='foreign')['value']
        assert state_of('deleg_wopen')['delivery_state'] == 'pending'
    finally:
        db.close()
