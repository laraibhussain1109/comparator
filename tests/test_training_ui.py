"""Real dataset labels remain visible when a housing cannot enter training."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import time

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

from src.ui.dialogs import TrainingDialog
from src.utils.config import deep_merge
from test_smoke_training import make_service


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def capture(service, index, label="GOOD", part_id="LEMON", session=""):
    image = np.random.default_rng(index).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    return service.capture_sample(image, "inner_face", label, part_id, session)


def test_imported_good_images_stay_good_when_shared_housing_has_ng(app, tmp_path, monkeypatch):
    service = make_service(tmp_path)
    samples = [capture(service, 1), capture(service, 2), capture(service, 3, "NG")]
    originals = {Path(sample["path"]): Path(sample["path"]).read_bytes() for sample in samples}
    metadata = {path.with_suffix(path.suffix + ".json"): path.with_suffix(path.suffix + ".json").read_bytes()
                for path in originals}
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(str(args[-1])))
    dialog = TrainingDialog(service)

    statistics = service.dataset_stats()
    assert statistics["good_images"] == 2
    assert statistics["ng_images"] == 1
    assert statistics["good_parts"] == 0
    assert statistics["ng_parts"] == 1
    assert "LEMON" in dialog.readiness.text()
    assert "NG" in dialog.readiness.text()
    assert "GOOD" in dialog.readiness.text()
    assert warnings == []

    dialog.start_job()
    deadline = time.monotonic() + 5
    while dialog.busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.003)
    assert not dialog.busy
    assert len(warnings) == 1
    assert "LEMON" in warnings[0]
    assert "3 independent GOOD" in warnings[0]
    assert [sample["label"] for sample in service.samples()].count(0) == 2
    assert all(path.read_bytes() == original for path, original in originals.items())
    assert all(path.read_bytes() == original for path, original in metadata.items())
    dialog.close()


def test_saved_session_grouping_is_used_in_training_summary(app, tmp_path):
    service = make_service(tmp_path)
    service.config.setdefault("dataset", {})["group_sessions"] = True
    for index in range(3):
        capture(service, index, part_id=f"actual-housing-{index}", session="shared-session")
    dialog = TrainingDialog(service)
    readiness = service.dataset_stats()["training_readiness"]
    assert readiness["parts"] == 3
    assert readiness["eligible_good_groups"] == 1
    assert dialog.group_sessions.isChecked()
    assert "session" in dialog.readiness.text().lower()
    dialog.close()


class ModeTrainingService:
    """Worker fixture for saved-model mode activation, without neural fitting."""
    def __init__(self, *, backend="hybrid", unlocked=True):
        self.config = {"product": "housing", "models": {"backend": backend},
                       "dataset": {"split": [.7, .15, .15]}}
        self.technician_unlocked = unlocked
        self.calls = []
        self.failure = None
        self.activation_failure = None
        self.lock_after_training = False

    def technician_configured(self):
        return True

    def dataset_stats(self):
        return {"images": 8, "parts": 4}

    def configure(self, settings):
        assert self.technician_unlocked
        self.calls.append(("configure", settings))
        self.config = deep_merge(self.config, settings)

    def set_mode(self, backend):
        self.calls.append(("set_mode", backend))
        if self.activation_failure:
            raise self.activation_failure
        self.configure({"models": {"backend": backend}})

    def train(self, view, backend, progress):
        self.calls.append(("train", view, backend, self.config["models"]["backend"]))
        if self.failure:
            raise self.failure
        if self.lock_after_training:
            self.technician_unlocked = False
        return {"view": view, "backend": "efficientad" if backend == "hybrid" else backend,
                "version": "saved-fixture-v1", "artifact_path": "/saved/models/" + view + "/saved-fixture-v1"}

    def evaluate(self, view, backend, progress):
        self.calls.append(("evaluate", view, backend))
        return {"parts": {"count": 3}}


def finish_worker(app, dialog):
    deadline = time.monotonic() + 3
    while dialog.busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.003)
    assert not dialog.busy


def test_successful_training_activates_selected_mode_only_after_model_save(app):
    service = ModeTrainingService()
    dialog = TrainingDialog(service)
    dialog.backend.setCurrentIndex(dialog.backend.findData("geometry"))
    changed = []
    dialog.models_changed.connect(lambda: changed.append(True))

    dialog.start_job()
    finish_worker(app, dialog)

    assert ("train", "inner_face", "geometry", "hybrid") in service.calls
    assert service.config["models"]["backend"] == "geometry"
    training_call = next(index for index, call in enumerate(service.calls) if call[0] == "train")
    activation_call = next(index for index, call in enumerate(service.calls) if call[0] == "set_mode")
    assert training_call < activation_call
    assert "backend" not in service.calls[0][1]["models"]
    assert "Face A" in dialog.footer.text()
    assert "Active inspection mode: Geometry" in dialog.footer.text()
    assert "/saved/models/inner_face/saved-fixture-v1" in dialog.footer.text()
    assert dialog.export.isEnabled()
    assert changed == [True]
    dialog.close()


def test_failed_training_never_switches_active_mode(app, monkeypatch):
    service = ModeTrainingService()
    service.failure = ValueError("controlled fitting failure")
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(str(args[-1])))
    dialog = TrainingDialog(service)
    dialog.backend.setCurrentIndex(dialog.backend.findData("geometry"))
    changed = []
    dialog.models_changed.connect(lambda: changed.append(True))

    dialog.start_job()
    finish_worker(app, dialog)

    assert service.config["models"]["backend"] == "hybrid"
    assert not any(call[0] == "set_mode" for call in service.calls)
    assert changed == []
    assert not dialog.export.isEnabled()
    assert "controlled fitting failure" in dialog.footer.text()
    assert len(warnings) == 1
    dialog.close()


def test_locked_operator_trains_commissioned_mode_without_settings_changes(app):
    service = ModeTrainingService(unlocked=False)
    dialog = TrainingDialog(service)
    assert not dialog.backend.isEnabled()
    # A disabled control's value cannot override the commissioned mode.
    dialog.backend.setCurrentIndex(dialog.backend.findData("geometry"))

    dialog.start_job()
    finish_worker(app, dialog)

    assert service.calls == [("train", "inner_face", "hybrid", "hybrid")]
    assert service.config["models"]["backend"] == "hybrid"
    assert dialog.backend.currentData() == "hybrid"
    assert not dialog.backend.isEnabled()
    assert "Active inspection mode: Hybrid" in dialog.footer.text()
    assert dialog.export.isEnabled()
    dialog.close()


def test_requested_hybrid_stays_hybrid_when_saved_backend_is_efficientad(app):
    service = ModeTrainingService(backend="geometry")
    dialog = TrainingDialog(service)
    dialog.backend.setCurrentIndex(dialog.backend.findData("hybrid"))

    dialog.start_job()
    finish_worker(app, dialog)

    assert dialog.report["backend"] == "efficientad"
    assert service.config["models"]["backend"] == "hybrid"
    assert ("set_mode", "hybrid") in service.calls
    assert ("set_mode", "efficientad") not in service.calls
    assert "Active inspection mode: Hybrid" in dialog.footer.text()
    dialog.close()


def test_activation_failure_preserves_successful_saved_model_report(app):
    service = ModeTrainingService()
    service.activation_failure = OSError("controlled settings write failure")
    dialog = TrainingDialog(service)
    dialog.backend.setCurrentIndex(dialog.backend.findData("geometry"))
    changed = []
    dialog.models_changed.connect(lambda: changed.append(True))

    dialog.start_job()
    finish_worker(app, dialog)

    assert dialog.report["version"] == "saved-fixture-v1"
    assert dialog.export.isEnabled()
    assert service.config["models"]["backend"] == "hybrid"
    assert "Training completed" in dialog.footer.text()
    assert "activating Geometry failed" in dialog.footer.text()
    assert "controlled settings write failure" in dialog.footer.text()
    assert "Active inspection mode: Hybrid" in dialog.footer.text()
    assert changed == [True]
    dialog.close()


def test_lost_technician_access_after_training_does_not_change_mode(app):
    service = ModeTrainingService()
    service.lock_after_training = True
    dialog = TrainingDialog(service)
    dialog.backend.setCurrentIndex(dialog.backend.findData("geometry"))

    dialog.start_job()
    finish_worker(app, dialog)

    assert dialog.export.isEnabled()
    assert service.config["models"]["backend"] == "hybrid"
    assert not any(call[0] == "set_mode" for call in service.calls)
    assert "Unlock Technician Access" in dialog.footer.text()
    assert not dialog.backend.isEnabled()
    dialog.close()


def test_evaluation_mode_selection_remains_available_without_activating_it(app):
    service = ModeTrainingService(unlocked=False)
    dialog = TrainingDialog(service, evaluate=True)
    assert dialog.backend.isEnabled()
    dialog.backend.setCurrentIndex(dialog.backend.findData("geometry"))
    changed = []
    dialog.models_changed.connect(lambda: changed.append(True))

    dialog.start_job()
    finish_worker(app, dialog)

    assert service.calls == [("evaluate", "all", "geometry")]
    assert service.config["models"]["backend"] == "hybrid"
    assert "Evaluation completed" in dialog.footer.text()
    assert changed == []
    assert dialog.backend.isEnabled()
    dialog.close()
