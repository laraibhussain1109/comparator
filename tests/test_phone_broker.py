"""Queued phone taps cannot drift onto a different part, face or capture."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.remote.broker import PhoneControlBroker


@pytest.fixture
def broker():
    control = PhoneControlBroker().start()
    control.publish(dict(mode='manual', event_id='event-1', part_id='part-1',
                         view='inner_face', can_submit=True, reason='', _context_key=('capture-1', 0)))
    return control


def command(broker, request_id='tap-1', verdict='PASS'):
    return dict(request_id=request_id, verdict=verdict, context_id=broker.snapshot()[1]['context_id'])


def test_acknowledgment_waits_for_pc_and_retries_are_idempotent(broker):
    payload = command(broker)
    assert broker.submit(payload)[0] == 202
    assert not broker.snapshot()[1]['can_submit']
    assert broker.submit(payload)[0] == 202
    assert broker.take_next() == payload
    assert broker.take_next() is None
    assert broker.command_status('tap-1')[1]['status'] == 'pending'
    result = broker.complete('tap-1', True, verdict='PASS')
    assert result == (200, dict(request_id='tap-1', verdict='PASS', status='applied', display_decision='OK'))
    assert broker.submit(payload) == result
    assert broker.take_next() is None
    assert broker.snapshot()[1]['last_command'] == result[1]


@pytest.mark.parametrize('change', [dict(view='ribbed_face'), dict(event_id='event-2'),
                                    dict(_context_key=('capture-2', 0)), dict(can_submit=False, reason='Automatic mode')])
def test_queued_decision_rejected_if_inspection_changes(broker, change):
    payload = command(broker)
    broker.submit(payload)
    state = dict(mode='manual', event_id='event-1', part_id='part-1', view='inner_face',
                 can_submit=True, reason='', _context_key=('capture-1', 0))
    state.update(change)
    broker.publish(state)
    assert broker.take_next() is None
    code, result = broker.command_status('tap-1')
    assert code == 409 and result['status'] == 'rejected'
    assert broker.submit(payload)[0] == 409


def test_same_state_keeps_context_and_public_snapshots_cannot_mutate_broker(broker):
    previous = broker.snapshot()[1]
    broker.publish(dict(mode='manual', event_id='event-1', part_id='part-1', view='inner_face',
                        can_submit=True, reason='', _context_key=('capture-1', 0)))
    assert broker.snapshot()[1]['context_id'] == previous['context_id']
    assert '_context_key' not in previous
    previous['can_submit'] = False
    assert broker.snapshot()[1]['can_submit']


def test_parallel_requests_queue_only_one_decision(broker):
    payloads = [command(broker, 'tap-a'), command(broker, 'tap-b')]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(broker.submit, payloads))
    assert sorted(code for code, _ in results) == [202, 409]
    accepted = broker.take_next()
    assert accepted['request_id'] in {'tap-a', 'tap-b'}
    assert broker.take_next() is None
    conflict = dict(accepted, verdict='FAIL')
    assert broker.submit(conflict)[0] == 409


def test_late_tap_expires_but_claimed_work_waits_for_real_result():
    now = [100.]
    broker = PhoneControlBroker(clock=lambda: now[0], command_ttl=2).start()
    broker.publish(dict(can_submit=True))
    broker.submit(command(broker))
    now[0] += 3
    assert broker.take_next() is None
    assert broker.command_status('tap-1')[0] == 410
    broker.submit(command(broker, 'tap-2'))
    broker.take_next()
    now[0] += 100
    assert broker.command_status('tap-2')[0] == 202
    assert broker.complete('tap-2', True)[0] == 200


def test_stop_rejects_unclaimed_taps_and_restart_waits_for_fresh_pc_state(broker):
    old = command(broker)
    broker.submit(old)
    broker.stop()
    assert broker.command_status('tap-1')[0] == 503
    assert not broker.snapshot()[1]['can_submit']
    broker.start()
    assert broker.submit(old)[0] == 409
    assert not broker.snapshot()[1]['can_submit']
    broker.publish(dict(can_submit=True))
    assert broker.submit(command(broker, 'new-tap'))[0] == 202


def test_pc_failure_is_reported_without_a_successful_verdict(broker):
    broker.submit(command(broker))
    broker.take_next()
    code, result = broker.complete('tap-1', False, error='View became locked')
    assert code == 409 and result['status'] == 'rejected'
    assert result['error'] == 'View became locked'
    assert 'display_decision' not in result


@pytest.mark.parametrize('payload', [None, [], {}, {'request_id': '../secret'},
    {'request_id': 'a', 'context_id': 'current', 'verdict': 'OK'},
    {'request_id': 'a', 'context_id': '', 'verdict': 'PASS'}])
def test_invalid_commands_never_reach_the_ui(broker, payload):
    assert broker.submit(payload)[0] == 400
    assert broker.take_next() is None
