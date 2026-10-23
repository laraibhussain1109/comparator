"""Dataset UI registration and repair tests using explicit physical identities."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from copy import deepcopy
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QItemSelectionModel, QTimer
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.ui.dialogs import DatasetDialog


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_until(app, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.003)
    assert predicate()


def select_rows(dialog, *rows):
    selection = dialog.table.selectionModel()
    selection.clearSelection()
    for row in rows:
        selection.select(dialog.table.model().index(row, 0),
                         QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
    if rows:
        dialog.table.setCurrentCell(rows[-1], 0, QItemSelectionModel.SelectionFlag.NoUpdate)


class PendingService:
    def __init__(self, directory):
        self.directory = directory
        self.managed = [{"id": "previous-valid", "path": str(directory / "previous.jpg"),
                         "part_id": "previous-part", "view": "inner_face", "label": 0,
                         "metadata": {"session_id": "previous-session"}}]
        self.pending = [
            {"path": str(directory / "capture-one.jpg"), "view": "inner_face", "label": "GOOD",
             "status": "unassigned", "reason": "Physical part metadata is missing"},
            {"path": str(directory / "capture-two.jpg"), "view": "inner_face", "label": "GOOD",
             "status": "repair_required", "reason": "The sidecar JSON is empty"}]
        for name in ["previous.jpg", "capture-one.jpg", "capture-two.jpg"]:
            assert cv2.imwrite(str(directory / name), np.zeros((12, 18, 3), np.uint8))
        self.calls = []
        self.fail_names = set()
        self.delay = 0
        self.inventory_error_after_write = False

    def samples(self):
        if self.inventory_error_after_write and any(call[0] in {"repair", "import"} for call in self.calls):
            raise ValueError("Separate checksum conflict requires review")
        return deepcopy(self.managed)

    def dataset_stats(self):
        return {"images": len(self.managed), "pending_count": len(self.pending)}

    def pending_samples(self):
        return deepcopy(self.pending)

    def repair_sample(self, path, view, label, part_id, session_id=""):
        return self._save("repair", path, view, label, part_id, session_id)

    def import_sample(self, path, view, label, part_id, session_id=""):
        return self._save("import", path, view, label, part_id, session_id)

    def _save(self, action, path, view, label, part_id, session_id):
        self.calls.append((action, path, view, label, part_id, session_id, threading.get_ident()))
        time.sleep(self.delay)
        if Path(path).name in self.fail_names:
            raise ValueError("Selected file failed integrity validation")
        assigned = self.directory / f"{part_id}__{Path(path).stem}__assigned.jpg"
        sample = {"id": f"saved-{len(self.managed)}", "path": str(assigned), "view": view,
                  "label": 0 if label == "GOOD" else 1, "part_id": part_id,
                  "metadata": {"session_id": session_id, "original_filename": Path(path).name}}
        self.managed.append(sample)
        self.pending = [entry for entry in self.pending if entry["path"] != path]
        return deepcopy(sample)

    def remove_sample(self, sample_id):
        self.calls.append(("remove", sample_id))
        self.managed = [sample for sample in self.managed if sample["id"] != sample_id]


def create_dialog(app, tmp_path):
    service = PendingService(tmp_path)
    dialog = DatasetDialog(service)
    wait_until(app, lambda: not dialog.busy)
    return dialog, service


def test_pending_photos_open_as_registration_and_repair_rows_without_modal(app, tmp_path, monkeypatch):
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: messages.append(args))
    dialog, service = create_dialog(app, tmp_path)
    assert messages == []
    assert dialog.table.rowCount() == 3
    assert [dialog.table.item(row, 0).text() for row in range(3)] == ["Registered", "Needs Registration", "Needs Repair"]
    assert "2 photos need registration or repair" in dialog.footer.text()
    assert dialog.part_id.text() == ""
    dialog.close()


def test_pending_selection_never_passes_filename_as_reject_sample_id(app, tmp_path):
    dialog, service = create_dialog(app, tmp_path)
    select_rows(dialog, 1)
    assert dialog.repair_button.isEnabled()
    assert not dialog.remove_button.isEnabled()
    dialog.remove_selected()
    assert service.calls == []
    select_rows(dialog, 0, 1)
    assert not dialog.repair_button.isEnabled()
    assert not dialog.remove_button.isEnabled()
    dialog.repair_selected()
    assert service.calls == []
    dialog.close()


def test_repair_requires_operator_physical_identity_without_inference_from_filename(app, tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    dialog, service = create_dialog(app, tmp_path)
    select_rows(dialog, 1)
    dialog.repair_selected()
    assert not service.calls
    assert "physical part ID is required" in warnings[-1]
    assert dialog.part_id.text() == ""
    dialog.close()


def test_repair_batch_preserves_valid_samples_and_reports_each_assigned_filename(app, tmp_path):
    dialog, service = create_dialog(app, tmp_path)
    service.fail_names.add("capture-two.jpg")
    dialog.part_id.setText("housing-42")
    dialog.session_id.setText("shift-7")
    dialog.view.setCurrentIndex(dialog.view.findData("ribbed_face"))
    dialog.label.setCurrentText("NG")
    select_rows(dialog, 1, 2)
    dialog.repair_selected()
    wait_until(app, lambda: not dialog.busy)
    assert len(service.calls) == 2
    assert all(call[2:6] == ("ribbed_face", "NG", "housing-42", "shift-7") for call in service.calls)
    assert any(sample["id"] == "previous-valid" for sample in service.managed)
    assert len(dialog._samples) == 2 and len(dialog._pending) == 1
    report = dialog.batch_report.toPlainText()
    assert "1 saved / registered; 1 failed" in report
    assert "capture-one.jpg → housing-42__capture-one__assigned.jpg" in report
    assert "FAILED: capture-two.jpg" in report
    assert "integrity validation" in report
    assert dialog._samples[0]["part_id"] == "previous-part"
    dialog.close()


def test_import_processes_independent_file_failures_and_shows_mapping(app, tmp_path, monkeypatch):
    dialog, service = create_dialog(app, tmp_path)
    service.fail_names.add("external-bad.jpg")
    files = [str(tmp_path / "external-good.jpg"), str(tmp_path / "external-bad.jpg")]
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *args: (files, "Images"))
    dialog.part_id.setText("real-housing")
    dialog.import_images()
    wait_until(app, lambda: not dialog.busy)
    assert [call[0] for call in service.calls] == ["import", "import"]
    assert all(call[4] == "real-housing" for call in service.calls)
    assert "external-good.jpg → real-housing__external-good__assigned.jpg" in dialog.batch_report.toPlainText()
    assert "FAILED: external-bad.jpg" in dialog.batch_report.toPlainText()
    assert any(sample["id"] == "previous-valid" for sample in dialog._samples)
    dialog.close()


def test_successful_batch_outcomes_survive_separate_inventory_failure(app, tmp_path, monkeypatch):
    dialog, service = create_dialog(app, tmp_path)
    service.inventory_error_after_write = True
    dialog.part_id.setText("housing-42")
    select_rows(dialog, 1)
    dialog.repair_selected()
    wait_until(app, lambda: not dialog.busy)
    report = dialog.batch_report.toPlainText()
    assert "1 saved / registered; 0 failed" in report
    assert "capture-one.jpg → housing-42__capture-one__assigned.jpg" in report
    assert "Separate checksum conflict requires review" in report
    assert "resolve the reported integrity problem before training" in dialog.footer.text()
    dialog.close()


def test_repair_runs_off_gui_thread_and_preserves_event_loop(app, tmp_path):
    dialog, service = create_dialog(app, tmp_path)
    service.delay = .08
    dialog.part_id.setText("housing-42")
    select_rows(dialog, 1)
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start(5)
    dialog.repair_selected()
    wait_until(app, lambda: not dialog.busy)
    timer.stop()
    assert len(ticks) >= 3
    assert service.calls[0][-1] != threading.get_ident()
    dialog.close()


def test_provider_without_pending_api_remains_compatible(app):
    class OlderService:
        def samples(self):
            return []
        def dataset_stats(self):
            return {"images": 0}
    dialog = DatasetDialog(OlderService())
    wait_until(app, lambda: not dialog.busy)
    assert dialog.table.rowCount() == 0
    assert not dialog.repair_button.isEnabled()
    dialog.close()


def test_real_service_dialog_repairs_same_raw_and_broken_json_photos(app, tmp_path, monkeypatch):
    """Regression for opening Dataset Manager with legacy unmanaged/broken photos."""
    from src.service import NeuroIrisService
    config = {"product": "housing", "paths": {name: str(tmp_path / name) for name in ("dataset", "artifacts", "results", "runtime")}}
    service = NeuroIrisService(config)
    source = tmp_path / "valid-external.jpg"
    assert cv2.imwrite(str(source), np.random.default_rng(1).integers(0, 255, (20, 24, 3), dtype=np.uint8))
    previous = service.import_sample(source, "inner_face", "GOOD", "previous-housing")
    folder = Path(config["paths"]["dataset"]) / "inner_face" / "GOOD"
    raw = folder / "same-raw-photo.jpg"
    broken = folder / "same-broken-photo.jpg"
    assert cv2.imwrite(str(raw), np.random.default_rng(2).integers(0, 255, (20, 24, 3), dtype=np.uint8))
    assert cv2.imwrite(str(broken), np.random.default_rng(3).integers(0, 255, (20, 24, 3), dtype=np.uint8))
    broken.with_suffix(".jpg.json").write_text("", encoding="utf-8")
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    dialog = DatasetDialog(service)
    wait_until(app, lambda: not dialog.busy)
    assert warnings == []
    assert len(dialog._samples) == 1 and len(dialog._pending) == 2
    dialog.part_id.setText("housing-42")
    pending_rows = [row for row, entry in enumerate(dialog._entries) if entry["pending"]]
    select_rows(dialog, *pending_rows)
    dialog.repair_selected()
    wait_until(app, lambda: not dialog.busy)
    assert warnings == []
    assert len(dialog._samples) == 3 and dialog._pending == []
    assert any(sample["id"] == previous["id"] for sample in dialog._samples)
    repaired = [sample for sample in dialog._samples if sample["part_id"] == "housing-42"]
    assert {sample["metadata"]["original_filename"] for sample in repaired} == {raw.name, broken.name}
    for sample in repaired:
        assert Path(sample["path"]).is_file()
        assert Path(sample["path"]).name in dialog.batch_report.toPlainText()
    assert not raw.exists() and not broken.exists()
    assert service.pending_samples() == []
    dialog.close()
