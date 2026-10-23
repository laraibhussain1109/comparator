"""Manual verdict keys remain scoped to deliberate, accountable view inspection."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import csv
import threading
import time

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QFileDialog, QLabel, QLineEdit

from src.service import NeuroIrisService
from src.ui.dialogs import HistoryDialog
from src.ui.neuroiris import MainWindow


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
        "product": "fixture",
        "paths": {name: tmp_path / name for name in ("dataset", "artifacts", "results", "runtime")},
        "views": {"required": ["inner_face", "ribbed_face"], "enabled": ["inner_face", "ribbed_face"]},
        "models": {"backend": "geometry"}, "quality": {},
    })
    monkeypatch.setattr(service, "diagnostics", lambda: {"camera": {"status": "stopped"}})
    monkeypatch.setattr(service, "model_readiness", lambda: {"ready": False, "views": [], "message": "No saved models"})
    monkeypatch.setattr(service, "_infer", lambda *args, **kwargs: pytest.fail("Manual inspection must not run inference"))
    widget = MainWindow(service)
    widget.show()
    widget.activateWindow()
    app.processEvents()
    yield widget
    wait_until(app, lambda: widget._thread is None)
    widget.close()
    app.processEvents()


def start_manual(window, app):
    window.inspection_mode.setCurrentIndex(window.inspection_mode.findData("manual"))
    assert window.verdict.text() == "READY", "Missing models must not block manual assessment"
    assert window.selected_view() == "inner_face"
    assert not window.view.model().item(window.view.findData("auto")).isEnabled()
    window.part_id.setText("manual-housing")
    window.begin_part()
    assert not window.inspection_mode.isEnabled()
    window.view.setFocus()
    app.processEvents()


def key(window, key_value, modifiers=Qt.KeyboardModifier.NoModifier):
    QTest.keyClick(window, key_value, modifiers)


@pytest.mark.parametrize("button,verdict,label,color", [
    (Qt.Key.Key_O, "PASS", "OK", "#00E676"),
    (Qt.Key.Key_R, "RECHECK", "View Recheck", "#ffca54"),
    (Qt.Key.Key_N, "FAIL", "NG", "#ff5252"),
])
def test_manual_keys_display_and_record_native_view_decisions(window, app, button, verdict, label, color, tmp_path, monkeypatch):
    start_manual(window, app)
    image = np.full((60, 90, 3), 110, np.uint8)
    window.analyze(image)
    key(window, button)
    wait_until(app, lambda: window._thread is None)
    assert window.verdict.text() == label
    assert color in window.verdict.styleSheet()
    assert window.service.session.decisions["inner_face"].verdict == verdict
    assert np.array_equal(window.service.frames["inner_face"], image)
    assert window.last_result["metrics"]["decision_source"] == "operator"
    assert window._manual_image is None
    assert not window.log.isVisible()
    assert not any("keyboard" in label.text().lower() for label in window.findChildren(QLabel))
    window.finish_part()
    wait_until(app, lambda: window._thread is None)
    expected_final = "NG" if verdict == "FAIL" else "View Recheck"
    assert window.verdict.text() == expected_final
    history = window.service.history()
    assert len(history) == 1 and history[0]["integrity_verified"]
    assert history[0]["inspection_mode"] == "manual"
    assert window.count_labels["total"].text() == "1"
    dialog = HistoryDialog(window.service, window)
    assert dialog.table.item(0, 1).text() == expected_final
    exported = tmp_path / "history.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(exported), "CSV (*.csv)"))
    dialog.export_csv()
    with exported.open(encoding="utf-8-sig", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["verdict"] == expected_final
    dialog.close()


def test_manual_both_faces_save_one_ok_part_and_lock_completed_view(window, app):
    start_manual(window, app)
    window.analyze(np.full((60, 90, 3), 60, np.uint8))
    key(window, Qt.Key.Key_O)
    wait_until(app, lambda: window._thread is None)
    # A second key cannot change an already accepted view.
    key(window, Qt.Key.Key_N)
    assert window.service.session.decisions["inner_face"].verdict == "PASS"
    window.view.setCurrentIndex(window.view.findData("ribbed_face"))
    window.analyze(np.full((60, 90, 3), 120, np.uint8))
    key(window, Qt.Key.Key_O, Qt.KeyboardModifier.ShiftModifier)
    wait_until(app, lambda: window._thread is None)
    window.finish_part()
    wait_until(app, lambda: window._thread is None)
    assert window.verdict.text() == "OK"
    assert window.count_labels["PASS"].text() == "1"
    history = window.service.history()
    assert len(history) == 1
    assert set(history[0]["images"]) == {"inner_face", "ribbed_face"}
    assert history[0]["final_decision"]["metrics"]["decision_source"] == "operator"
    assert window.inspection_mode.isEnabled()


def test_keys_ignore_automatic_mode_typing_and_dialog_focus(window, app):
    key(window, Qt.Key.Key_O)
    assert window.service.session is None
    start_manual(window, app)
    window.analyze(np.zeros((60, 90, 3), np.uint8))
    editor = QLineEdit(window.centralWidget())
    editor.show()
    editor.setFocus()
    app.processEvents()
    QTest.keyClicks(editor, "orn")
    assert editor.text() == "orn"
    assert window.service.session.decisions == {}
    dialog = QDialog(window)
    dialog.show()
    dialog.activateWindow()
    app.processEvents()
    key(window, Qt.Key.Key_O)
    assert window.service.session.decisions == {}
    dialog.close()
    editor.deleteLater()


def test_manual_ignores_modifiers_repeats_missing_images_and_wrong_view_image(window, app):
    start_manual(window, app)
    key(window, Qt.Key.Key_O)
    assert window.service.session.decisions == {}
    window.analyze(np.zeros((60, 90, 3), np.uint8))
    key(window, Qt.Key.Key_O, Qt.KeyboardModifier.ControlModifier)
    repeat = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_O, Qt.KeyboardModifier.NoModifier, "o", True)
    app.sendEvent(window, repeat)
    assert window.service.session.decisions == {}
    window.view.setCurrentIndex(window.view.findData("ribbed_face"))
    assert window._manual_image is None
    key(window, Qt.Key.Key_O)
    assert window.service.session.decisions == {}


def test_manual_keys_cannot_race_an_active_worker(window, app):
    start_manual(window, app)
    window.analyze(np.zeros((60, 90, 3), np.uint8))
    release = threading.Event()
    started = threading.Event()

    def operation(progress):
        started.set()
        assert release.wait(3)

    window.run_task("Controlled operation", operation, lambda result: None)
    wait_until(app, started.is_set)
    try:
        assert not window.inspection_mode.isEnabled()
        key(window, Qt.Key.Key_N)
        assert window.service.session.decisions == {}
    finally:
        release.set()
        wait_until(app, lambda: window._thread is None)


def test_manual_recheck_can_be_retaken_from_a_new_current_image(window, app):
    start_manual(window, app)
    window.analyze(np.full((60, 90, 3), 60, np.uint8))
    key(window, Qt.Key.Key_R)
    wait_until(app, lambda: window._thread is None)
    window.camera_running = True
    window._latest = np.full((60, 90, 3), 100, np.uint8)
    key(window, Qt.Key.Key_O)
    wait_until(app, lambda: window._thread is None)
    assert window.verdict.text() == "OK"
    assert window.service.session.attempts[-1]["action"] == "retake"
    assert np.all(window.service.frames["inner_face"] == 100)


def test_manual_capture_freezes_native_pixels_until_decision(window, app):
    start_manual(window, app)
    window.camera_running = True
    window._latest = np.full((60, 90, 3), 45, np.uint8)
    window.capture_view()
    window._latest[:] = 210
    key(window, Qt.Key.Key_O)
    wait_until(app, lambda: window._thread is None)
    assert np.all(window.service.frames["inner_face"] == 45)
    window.view.setCurrentIndex(window.view.findData("ribbed_face"))
    assert not window._showing_result
    assert np.all(window.canvas.image == 210)
