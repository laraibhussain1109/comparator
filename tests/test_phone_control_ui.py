"""Phone commands cross the GUI thread only after current inspection checks."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
import secrets

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QDialog, QPushButton, QWidget

from src.service import NeuroIrisService
from src.ui.neuroiris import MainWindow
from src.ui.dialogs import TechnicalDialog
from src.ui.phone_control import PhoneControlDialog, pairing_qr_image


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_until(app, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert predicate()


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    service = NeuroIrisService({
        "product": "phone_fixture",
        "paths": {name: tmp_path / name for name in ("dataset", "artifacts", "results", "runtime")},
        "views": {"required": ["inner_face", "ribbed_face"], "enabled": ["inner_face", "ribbed_face"]},
        "models": {"backend": "geometry"}, "quality": {},
    })
    monkeypatch.setattr(service, "diagnostics", lambda: {"camera": {"status": "stopped"}})
    monkeypatch.setattr(service, "model_readiness", lambda: {"ready": False, "views": [], "message": "No saved models"})
    monkeypatch.setattr(service, "_infer", lambda *args, **kwargs: pytest.fail("Phone decisions must not run inference"))
    widget = MainWindow(service)
    widget._phone_timer.stop()
    widget.show()
    widget.activateWindow()
    app.processEvents()
    yield widget
    wait_until(app, lambda: widget._thread is None)
    widget.close()
    app.processEvents()


def prepare_manual(window):
    window.inspection_mode.setCurrentIndex(window.inspection_mode.findData("manual"))
    window.part_id.setText("phone-housing")
    window.begin_part()
    window.analyze(np.full((60, 90, 3), 45, np.uint8))
    window.phone_broker.start()
    return window.publish_phone_state()


def queue(window, request_id="phone-tap-1", verdict="PASS"):
    state = window.publish_phone_state()
    payload = {"request_id": request_id, "context_id": state["context_id"], "verdict": verdict}
    code, command = window.phone_broker.submit(payload)
    assert code == 202 and command["status"] == "pending"
    return payload


@pytest.mark.parametrize("verdict,display", [("PASS", "OK"), ("RECHECK", "View Recheck"), ("FAIL", "NG")])
def test_phone_verdict_is_acknowledged_only_after_gui_service_success(window, app, verdict, display):
    prepare_manual(window)
    payload = queue(window, verdict=verdict)
    assert window.service.session.decisions == {}
    window.poll_phone_control()
    wait_until(app, lambda: window._thread is None)
    code, command = window.phone_broker.command_status(payload["request_id"])
    assert code == 200 and command["status"] == "applied"
    assert command["display_decision"] == display
    assert window.verdict.text() == display
    assert window.service.session.decisions["inner_face"].verdict == verdict
    assert window.last_result["metrics"]["decision_source"] == "operator"
    # Retrying an HTTP request cannot add a second inspection attempt.
    assert window.phone_broker.submit(payload)[1]["status"] == "applied"
    window.poll_phone_control()
    assert len(window.service.session.attempts) == 1


@pytest.mark.parametrize("change", ["view", "capture", "mode", "dialog"])
def test_queued_phone_command_is_rejected_after_pc_context_changes(window, app, change):
    prepare_manual(window)
    payload = queue(window)
    if change == "view":
        window.view.setCurrentIndex(window.view.findData("ribbed_face"))
    elif change == "capture":
        window.analyze(np.full((60, 90, 3), 99, np.uint8))
    elif change == "mode":
        # A programmatic stale view of the mode must also fail before dispatch.
        window.inspection_mode.blockSignals(True)
        window.inspection_mode.setCurrentIndex(window.inspection_mode.findData("automatic"))
        window.inspection_mode.blockSignals(False)
    else:
        dialog = QDialog(window)
        window._open_dialog(dialog)
        app.processEvents()
    window.poll_phone_control()
    code, command = window.phone_broker.command_status(payload["request_id"])
    assert code == 409 and command["status"] == "rejected"
    assert window.service.session.decisions == {}


def test_phone_control_requires_manual_part_image_and_unlocked_face(window, app):
    assert window.phone_server is None
    window.phone_broker.start()
    assert not window.publish_phone_state()["can_submit"]
    window.inspection_mode.setCurrentIndex(window.inspection_mode.findData("manual"))
    assert not window.publish_phone_state()["can_submit"]
    window.part_id.setText("needs-image")
    window.begin_part()
    assert not window.publish_phone_state()["can_submit"]
    window.analyze(np.zeros((60, 90, 3), np.uint8))
    assert window.publish_phone_state()["can_submit"]
    queue(window)
    window.poll_phone_control()
    wait_until(app, lambda: window._thread is None)
    assert not window.publish_phone_state()["can_submit"]
    window.finish_part()
    wait_until(app, lambda: window._thread is None)
    assert not window.publish_phone_state()["can_submit"]


def test_phone_controls_work_while_main_window_is_unfocused(window, app):
    prepare_manual(window)
    other = QWidget()
    other.show()
    other.activateWindow()
    app.processEvents()
    try:
        assert not window.isActiveWindow()
        queue(window)
        window.poll_phone_control()
        wait_until(app, lambda: window._thread is None)
        assert window.verdict.text() == "OK"
    finally:
        other.close()


def test_phone_service_failure_is_rejected_and_reported_on_pc(window, app, monkeypatch):
    prepare_manual(window)
    errors = []
    monkeypatch.setattr(window, "_error", errors.append)
    def fail(*args):
        raise RuntimeError("The current event is locked")
    monkeypatch.setattr(window.service, "record_manual_decision", fail)
    payload = queue(window)
    window.poll_phone_control()
    wait_until(app, lambda: window._thread is None)
    code, command = window.phone_broker.command_status(payload["request_id"])
    assert code == 409 and command["status"] == "rejected"
    assert "event is locked" in command["error"]
    assert errors and window.service.session.decisions == {}


def test_queued_phone_command_cannot_race_an_active_pc_worker(window, app):
    prepare_manual(window)
    payload = queue(window)
    release = threading.Event()
    started = threading.Event()
    def operation(progress):
        started.set()
        assert release.wait(3)
    window.run_task("Controlled operation", operation, lambda result: None)
    wait_until(app, started.is_set)
    try:
        window.poll_phone_control()
        assert window.phone_broker.command_status(payload["request_id"])[1]["status"] == "rejected"
        assert window.service.session.decisions == {}
    finally:
        release.set()
        wait_until(app, lambda: window._thread is None)


def test_phone_setup_starts_explicitly_rotates_pairing_and_stops_on_close(window, app, monkeypatch):
    # Socket transport is covered separately; this test exercises the real GUI
    # lifecycle and broker without requiring network permission.
    class FakeServer:
        def __init__(self, broker, host, port):
            self.broker, self.port = broker, port or 54321
            self.running, self.token = False, None
        def start(self):
            self.running, self.token = True, secrets.token_urlsafe(32)
            self.broker.start()
            return self
        def stop(self):
            self.running, self.token = False, None
            self.broker.stop()
        def pairing_url(self, address):
            return f"http://{address}:{self.port}/#token={self.token}"
    monkeypatch.setattr("src.remote.server.PhoneControlServer", FakeServer)
    assert window.phone_server is None
    assert not any("Phone Control" in button.text()
                   for button in window.centralWidget().findChildren(QPushButton))
    with monkeypatch.context() as locked:
        locked.setattr(window, "ensure_technician", lambda: False)
        window.open_phone_control()
        assert not window._dialogs
    window.service.set_technician_pin("local-phone-test-123")
    assert window.service.unlock_technician("local-phone-test-123")
    window.start_phone_control(0)
    first = window.phone_server
    token = first.token
    window.open_technical()
    app.processEvents()
    settings = next(dialog for dialog in window._dialogs if isinstance(dialog, TechnicalDialog))
    settings.phone_control.click()
    wait_until(app, lambda: any(isinstance(dialog, PhoneControlDialog) for dialog in window._dialogs))
    assert not any(isinstance(dialog, TechnicalDialog) and dialog.isVisible() for dialog in window._dialogs)
    dialog = next(dialog for dialog in window._dialogs if isinstance(dialog, PhoneControlDialog))
    assert first.running and not dialog.start.isEnabled()
    dialog.address.setEditText("192.168.1.50")
    assert f":{first.port}/" in dialog.link.text()
    assert token in dialog.link.text()
    assert dialog.qrcode_image is not None
    dialog.copy_link()
    assert QApplication.clipboard().text() == dialog.link.text()
    dialog.stop_server()
    assert not first.running and window.phone_server is None
    assert not dialog.link.text() and not dialog.copy.isEnabled()
    assert dialog.qrcode_image is None and dialog.qrcode.isHidden()
    window.start_phone_control(0)
    second = window.phone_server
    assert second.token != token
    window.close()
    app.processEvents()
    assert not second.running


def test_pairing_qr_decodes_to_exact_synthetic_link():
    link = "http://192.168.1.50:8765/#token=synthetic-pairing-token-only"
    pixels = pairing_qr_image(link)
    assert pixels is not None
    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(pixels)
    assert decoded == link
