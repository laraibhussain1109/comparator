"""UI software tests: provenance, worker responsiveness and coordinate mapping."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
from copy import deepcopy

import numpy as np
import pytest
from PyQt6.QtCore import QPointF, QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox

from src.ui.neuroiris import MainWindow
from src.ui.dialogs import (CameraDialog, DatasetDialog, HistoryDialog,
                           ModelSettingsDialog, TechnicalDialog, TrainingDialog)
from src.ui.roi_editor import ROICanvas, ROIEditor


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class FakeService:
    def __init__(self):
        self.config = {"product": "aluminium_housing", "camera": {}, "models": {"backend": "geometry"},
                       "views": {"required": ["inner_face", "ribbed_face"]}, "dataset": {"split": [.7, .15, .15]}}
        self.rows = []
        self.rois = {}
        self.calls = []
        self.session = None

    def diagnostics(self):
        return {"camera": {"connected": False, "status": "stopped"}}

    def dataset_stats(self):
        return {"images": 0, "parts": 0}

    def samples(self, view=None):
        return []

    def history(self):
        return self.rows

    def preview(self):
        return {"image": np.zeros((100, 200, 3), np.uint8), "sequence": 1}

    def models(self):
        return []

    def stop_camera(self):
        self.calls.append("stop")

    def set_camera_control(self, name, value):
        return False

    def get_rois(self, view):
        return deepcopy(self.rois.get(view, []))

    def save_rois(self, view, definitions):
        self.rois[view] = deepcopy(definitions)

    def configure(self, settings):
        self.calls.append(("configure", settings))

    def begin_part(self, identity):
        self.session = {"part_id": identity, "status": "AWAITING_VIEWS", "views": {},
                        "missing_views": ["inner_face", "ribbed_face"]}
        return deepcopy(self.session)

    def inspect_live(self, view, identity):
        self.calls.append(("live", view, identity))
        return self._inspect(view)

    def inspect_frame(self, image, view, identity):
        self.calls.append(("file", view, identity))
        return self._inspect(view)

    def _inspect(self, view):
        self.session["views"][view] = {"verdict": "PASS"}
        self.session["missing_views"] = [v for v in ["inner_face", "ribbed_face"] if v not in self.session["views"]]
        return {"view": view, "verdict": "PASS", "reasons": ["Controlled fixture"],
                "marked_image": np.zeros((100, 200, 3), np.uint8), "session": deepcopy(self.session), "candidates": []}

    def finish_part(self):
        verdict = "RECHECK" if self.session["missing_views"] else "PASS"
        self.rows.append({"counted": True, "integrity_verified": True, "final_decision": {"verdict": verdict}})
        return {"verdict": verdict, "reasons": []}

    def cancel_part(self):
        return {"verdict": "CANCELLED"}


def wait_until(app, predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert predicate()


def make_window(app):
    service = FakeService()
    window = MainWindow(service)
    return window, service


def test_count_only_finalized_integrity_verified_parts(app):
    window, service = make_window(app)
    service.rows = [
        {"counted": True, "integrity_verified": True, "final_decision": {"verdict": "PASS"}},
        {"counted": False, "integrity_verified": True, "verdict": "FAIL"},
        {"counted": True, "integrity_verified": False, "verdict": "FAIL"},
        {"counted": True, "integrity_verified": True, "verdict": "RECHECK"},
    ]
    window.refresh_counts()
    assert window.count_labels["total"].text() == "2"
    assert window.count_labels["PASS"].text() == "1"
    assert window.count_labels["FAIL"].text() == "0"
    window.close()


def test_live_capture_uses_stable_service_path_and_one_final_event(app):
    window, service = make_window(app)
    window.part_id.setText("housing-7")
    window.begin_part()
    window.camera_running = True
    window._latest = np.zeros((100, 200, 3), np.uint8)
    window.view.setCurrentIndex(window.view.findData("inner_face"))
    window.capture_view()
    wait_until(app, lambda: window._thread is None)
    assert service.calls[-1] == ("live", "inner_face", "housing-7")
    assert window.count_labels["total"].text() == "0"
    assert "NEEDED" in window.coverage.text()
    window.finish_part()
    wait_until(app, lambda: window._thread is None)
    assert window.verdict.text() == "RECHECK"
    assert window.count_labels["total"].text() == "1"
    window.close()


def test_offline_image_uses_explicit_frame_path(app):
    window, service = make_window(app)
    window.part_id.setText("housing-8")
    window.begin_part()
    window.view.setCurrentIndex(window.view.findData("ribbed_face"))
    window.analyze(np.zeros((100, 200, 3), np.uint8))
    wait_until(app, lambda: window._thread is None)
    assert ("file", "ribbed_face", "housing-8") in service.calls
    window.close()


def test_background_operation_keeps_event_loop_responsive(app):
    window, service = make_window(app)
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)
    window.run_task("controlled worker", lambda progress: time.sleep(.08))
    wait_until(app, lambda: window._thread is None)
    timer.stop()
    assert len(ticks) >= 3
    window.close()


def test_dialogs_construct_for_string_product_and_no_models(app):
    window, service = make_window(app)
    dialogs = [DatasetDialog(service, window), TrainingDialog(service, window),
               TrainingDialog(service, window, evaluate=True), CameraDialog(service, window),
               ModelSettingsDialog(service, window), HistoryDialog(service, window),
               TechnicalDialog(service, {}, window)]
    for dialog in dialogs:
        wait_until(app, lambda dialog=dialog: not getattr(dialog, "busy", False))
        assert dialog.windowTitle()
        dialog.close()
    window.close()


def test_roi_normalization_accounts_for_letterboxing(app):
    canvas = ROICanvas()
    canvas.resize(500, 500)
    canvas.set_image(np.zeros((100, 200, 3), np.uint8))
    rect = canvas.image_rect()
    assert rect.x() == 0 and rect.y() == 125
    assert canvas._normalized(QPointF(250, 250)) == [.5, .5]
    assert canvas._normalized(QPointF(10, 10)) is None
    canvas.close()


def test_roi_masks_save_independently_by_view(app):
    service = FakeService()
    editor = ROIEditor(service, image=np.zeros((100, 200, 3), np.uint8))
    editor.add_region({"shape": "rectangle", "points": [[.1, .2], [.8, .9]]})
    assert editor.save()
    assert len(service.rois["inner_face"]) == 1
    editor.view.setCurrentIndex(editor.view.findData("ribbed_face"))
    assert editor.canvas.regions == []
    editor.exclude.setChecked(True)
    editor.add_region({"shape": "circle", "points": [[.5, .5], [.6, .5]]})
    assert editor.save()
    assert service.rois["ribbed_face"][0]["exclusion"] is True
    assert service.rois["inner_face"][0]["exclusion"] is False
    editor.close()


def test_roi_selected_fields_edit_real_definition(app):
    service = FakeService()
    editor = ROIEditor(service, image=np.zeros((100, 200, 3), np.uint8))
    editor.add_region({"shape": "polygon", "points": [[.1, .1], [.8, .1], [.5, .8]]})
    editor.exclude.setChecked(True)
    assert editor.canvas.regions[0]["exclusion"] is True
    editor.delete_selected()
    assert editor.canvas.regions == []
    editor.save()
    editor.close()


def test_evaluation_defaults_to_all_views_and_benchmark_is_real(app):
    service = FakeService()
    service.evaluate = lambda view, backend, progress: service.calls.append(("evaluate", view, backend)) or {"scope": "all views", "parts": {"n": 3}}
    service.benchmark = lambda view, progress: service.calls.append(("benchmark", view)) or {"models": ["geometry"]}
    dialog = TrainingDialog(service, evaluate=True)
    assert dialog.view.currentData() == "all"
    dialog.start_job()
    wait_until(app, lambda: not dialog.busy)
    assert ("evaluate", "all", "geometry") in service.calls
    assert dialog.report["scope"] == "all views"
    dialog.start_benchmark()
    wait_until(app, lambda: not dialog.busy)
    assert ("benchmark", "all") in service.calls
    assert dialog.report == {"models": ["geometry"]}
    dialog.close()


class GuardService(FakeService):
    def __init__(self):
        super().__init__()
        self.configured = False
        self.technician_unlocked = False
        self._pin = None

    def technician_configured(self):
        return self.configured

    def set_technician_pin(self, pin):
        if self.configured or len(pin) < 6:
            raise ValueError("Initial setup only; minimum 6 characters")
        self._pin = pin
        self.configured = True

    def unlock_technician(self, pin):
        self.technician_unlocked = self._pin == pin
        return self.technician_unlocked

    def lock_technician(self):
        self.technician_unlocked = False


def test_first_technician_action_creates_confirms_and_unlocks_local_pin(app, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog
    service = GuardService()
    window = MainWindow(service)
    replies = iter([("local-test-123", True), ("local-test-123", True)])
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: next(replies))
    assert window.ensure_technician()
    assert service.technician_configured() and service.technician_unlocked
    assert window.technician_access.text() == "Lock Technician"
    window.toggle_technician()
    assert not service.technician_unlocked
    window.close()


def test_cancelled_or_wrong_technician_unlock_cannot_open_advanced_dialog(app, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog
    service = GuardService()
    service.set_technician_pin("local-test-123")
    window = MainWindow(service)
    opened = []
    monkeypatch.setattr(window, "_open_dialog", lambda dialog: opened.append(dialog))
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("", False))
    window.open_roi()
    assert opened == []
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("wrong", True))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    window.open_models()
    assert opened == []
    assert not service.technician_unlocked
    window.close()


def test_locked_operator_can_train_saved_settings_without_configure(app):
    service = GuardService()
    service.set_technician_pin("local-test-123")
    service.train = lambda view, backend, progress: service.calls.append(("train", view, backend)) or {"trained": True}
    dialog = TrainingDialog(service)
    assert not dialog.steps.isEnabled()
    dialog.start_job()
    wait_until(app, lambda: not dialog.busy)
    assert ("train", "inner_face", "geometry") in service.calls
    assert not any(isinstance(call, tuple) and call[0] == "configure" for call in service.calls)
    assert not dialog.steps.isEnabled()
    dialog.close()


def test_repeatability_summary_uses_only_verified_counted_unique_events():
    from src.ui.dialogs import repeatability_summary
    def record(event, verdict, **kwargs):
        return {"part_id": "same-housing", "event_id": event, "counted": True,
                "integrity_verified": True, "required_views": ["inner_face", "ribbed_face"],
                "final_decision": {"verdict": verdict}, "views": {"inner_face": {"model_version": "v1"}}, **kwargs}
    rows = [record("a", "PASS"), record("b", "RECHECK"), record("b", "RECHECK"),
            record("c", "FAIL", integrity_verified=False), record("d", "FAIL", counted=False)]
    report = repeatability_summary(rows)
    assert report["repeated_sessions"] == 2
    assert report["parts_with_mixed_verdicts"] == 1
    assert report["groups"][0]["verdict_counts"] == {"PASS": 1, "FAIL": 0, "RECHECK": 1}
    assert "accuracy" not in report
