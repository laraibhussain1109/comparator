"""Real local HTTP taps update the real desktop/session path on the Qt thread."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from http.client import HTTPConnection
import json
import time

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from src.service import NeuroIrisService
from src.ui.neuroiris import MainWindow


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    service = NeuroIrisService({
        'product': 'http_fixture',
        'paths': {name: tmp_path / name for name in ('dataset', 'artifacts', 'results', 'runtime')},
        'views': {'required': ['inner_face', 'ribbed_face'], 'enabled': ['inner_face', 'ribbed_face']},
        'models': {'backend': 'geometry'}, 'quality': {},
    })
    monkeypatch.setattr(service, 'diagnostics', lambda: {'camera': {'status': 'stopped'}})
    monkeypatch.setattr(service, 'model_readiness', lambda: {'ready': False, 'views': [], 'message': 'No models'})
    monkeypatch.setattr(service, '_infer', lambda *args: pytest.fail('Phone verdict must not run inference'))
    widget = MainWindow(service)
    widget.show()
    widget.inspection_mode.setCurrentIndex(widget.inspection_mode.findData('manual'))
    widget.part_id.setText('wifi-housing-1')
    widget.begin_part()
    widget.analyze(np.full((41, 59, 3), 87, np.uint8))
    widget.start_phone_control(port=0)
    yield widget
    deadline = time.monotonic() + 3
    while widget._thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert widget._thread is None
    widget.close()
    app.processEvents()


def http(window, path, payload=None):
    server = window.phone_server
    connection = HTTPConnection('127.0.0.1', server.port, timeout=2)
    headers = {'Authorization': 'Bearer ' + server.token}
    if payload is not None:
        headers['Content-Type'] = 'application/json'
    try:
        connection.request('GET' if payload is None else 'POST', path,
                           body=json.dumps(payload) if payload is not None else None,
                           headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def settle(app, window, request_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        app.processEvents()
        status = window.phone_broker.command_status(request_id)
        if status[0] != 202 and window._thread is None:
            return status
        time.sleep(.005)
    pytest.fail('The PC never completed the phone command')


@pytest.mark.parametrize('verdict, label', [('PASS', 'OK'), ('RECHECK', 'View Recheck'), ('FAIL', 'NG')])
def test_http_verdict_updates_gui_and_persists_original_evidence(window, app, verdict, label):
    code, state = http(window, '/api/state')
    assert code == 200 and state['can_submit']
    assert state['part_id'] == 'wifi-housing-1'
    payload = dict(request_id='real-phone-tap', context_id=state['context_id'], verdict=verdict)
    code, queued = http(window, '/api/verdict', payload)
    assert code == 202 and queued['status'] == 'pending'
    assert window.service.session.decisions == {}
    assert settle(app, window, 'real-phone-tap')[0] == 200
    assert window.verdict.text() == label
    code, acknowledged = http(window, '/api/commands/real-phone-tap')
    assert code == 200 and acknowledged['display_decision'] == label
    # A retry with the same ID returns the original receipt without a second capture.
    assert http(window, '/api/verdict', payload)[0] == 200
    assert len(window.service.session.attempts) == 1
    window.service.finish_part()
    row = window.service.history()[0]
    assert row['integrity_verified'] and row['inspection_mode'] == 'manual'
    assert row['views']['inner_face']['metrics']['decision_source'] == 'operator'
    assert row['views']['inner_face']['verdict'] == verdict
    assert np.all(window.service.frames['inner_face'] == 87)


def test_http_tap_for_previous_face_never_changes_the_new_face(window, app):
    _, state = http(window, '/api/state')
    window.view.setCurrentIndex(window.view.findData('ribbed_face'))
    # Submit before the GUI's next status tick: the UI must recheck on delivery.
    code, _ = http(window, '/api/verdict', dict(request_id='old-face-tap',
                    context_id=state['context_id'], verdict='FAIL'))
    assert code == 202
    code, result = settle(app, window, 'old-face-tap')
    assert code == 409 and result['status'] == 'rejected'
    assert window.service.session.decisions == {}
