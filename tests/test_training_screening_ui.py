"""Training UI reports per-run exclusions without changing dataset identity."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from copy import deepcopy
import time

import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

from src.ui.dialogs import TrainingDialog


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_until(app, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.003)
    assert predicate()


def screening_report(run_id="new-run", *, status="completed", filename="<b>blurred-housing</b>.jpg"):
    return {"schema_version": 1, "run_id": run_id, "view": "inner_face", "backend": "geometry",
            "status": status, "candidate_images": 8, "accepted_images": 6, "excluded_images": 2,
            "accepted_sample_ids": ["valid-a", "valid-b"],
            "excluded": [{"sample_id": "blurred", "filename": filename,
                          "path": "/operator/dataset/" + filename, "part_id": "physical-part-7",
                          "view": "inner_face", "stage": "frame_quality",
                          "reasons": ["Sharpness 2.0 is below required 35.0"], "metrics": {"sharpness": 2.0}},
                         {"sample_id": "clipped", "filename": "overexposed.jpg", "path": "/operator/dataset/overexposed.jpg",
                          "part_id": "physical-part-8", "view": "inner_face", "stage": "exposure",
                          "reasons": ["Saturated pixel fraction exceeds limit"], "metrics": {"bright_fraction": .7}}],
            "report_path": "/operator/runtime/training-screening/new-run.json"}


class ScreeningService:
    def __init__(self):
        self.config = {"product": "housing", "models": {"backend": "geometry"}, "dataset": {"split": [.7, .15, .15]}}
        self.technician_unlocked = False
        self.latest = screening_report("previous-run", filename="previous-photo.jpg")
        self.failure = None
        self.change_report = True
        self.getter_error = False
        self.calls = []

    def technician_configured(self):
        return True

    def dataset_stats(self):
        return {"images": 8, "parts": 8}

    def training_screening(self, view):
        self.calls.append(("screening", view))
        if self.getter_error:
            raise OSError("Screening report cannot be read")
        return deepcopy(self.latest)

    def train(self, view, backend, progress):
        self.calls.append(("train", view, backend))
        if self.change_report:
            self.latest = screening_report(status="failed" if self.failure else "completed")
            if self.failure:
                self.latest["error"] = self.failure
        progress(30, "Screening", "Excluded <b>blurred-housing</b>.jpg: insufficient focus")
        if self.failure:
            raise ValueError(self.failure)
        return {"model_version": "fixture-v1", "training_screening": deepcopy(self.latest)}


def test_completed_training_shows_counts_filenames_reasons_and_saved_report(app):
    service = ScreeningService()
    dialog = TrainingDialog(service)
    assert "previous run" in dialog.screening_heading.text()
    assert "previous-photo.jpg" in dialog.screening_output.toPlainText()
    dialog.start_job()
    wait_until(app, lambda: not dialog.busy)
    assert "this training attempt" in dialog.screening_heading.text()
    text = dialog.screening_output.toPlainText()
    assert "Candidate images: 8 · Accepted: 6 · Excluded: 2" in text
    assert "<b>blurred-housing</b>.jpg" in text
    assert "overexposed.jpg" in text
    assert "Sharpness 2.0 is below required 35.0" in text
    assert "Saturated pixel fraction exceeds limit" in text
    assert "Original files and labels are retained" in text
    assert "Report path: /operator/runtime/training-screening/new-run.json" in text
    assert dialog.screening_output.isReadOnly()
    assert dialog.export.isEnabled()
    assert not dialog.steps.isEnabled()
    dialog.close()


def test_screening_and_progress_filenames_are_literal_plain_text(app):
    service = ScreeningService()
    dialog = TrainingDialog(service)
    dialog.start_job()
    wait_until(app, lambda: not dialog.busy)
    assert "Excluded <b>blurred-housing</b>.jpg" in dialog.output.toPlainText()
    assert "<b>blurred-housing</b>.jpg" in dialog.screening_output.toPlainText()
    assert "&lt;b&gt;blurred-housing&lt;/b&gt;" in dialog.output.toHtml()
    dialog.close()


def test_failed_training_fetches_current_exclusions_and_preserves_real_error(app, monkeypatch):
    service = ScreeningService()
    service.failure = "Only two independent GOOD parts remain after screening"
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    dialog = TrainingDialog(service)
    dialog.start_job()
    wait_until(app, lambda: not dialog.busy)
    assert "this training attempt" in dialog.screening_heading.text()
    assert "Status: failed" in dialog.screening_output.toPlainText()
    assert service.failure in dialog.screening_output.toPlainText()
    assert "overexposed.jpg" in dialog.screening_output.toPlainText()
    assert service.failure in dialog.output.toPlainText()
    assert service.failure in dialog.footer.text()
    assert warnings and service.failure in warnings[-1]
    assert not dialog.export.isEnabled()
    assert not dialog.steps.isEnabled()
    dialog.close()


def test_previous_report_is_not_attributed_to_attempt_without_new_run(app, monkeypatch):
    service = ScreeningService()
    service.failure = "Training stopped before creating a screening report"
    service.change_report = False
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    dialog = TrainingDialog(service)
    dialog.start_job()
    wait_until(app, lambda: not dialog.busy)
    assert "Previous run" in dialog.screening_heading.text()
    assert "has not produced a new screening report" in dialog.screening_heading.text()
    assert "previous-photo.jpg" in dialog.screening_output.toPlainText()
    assert service.failure in dialog.footer.text()
    dialog.close()


def test_reporting_error_does_not_hide_training_error_or_progress(app, monkeypatch):
    service = ScreeningService()
    service.getter_error = True
    service.failure = "No usable training image remains"
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    dialog = TrainingDialog(service)
    dialog.start_job()
    wait_until(app, lambda: not dialog.busy)
    assert "could not be loaded" in dialog.screening_heading.text()
    assert "Screening report cannot be read" in dialog.screening_output.toPlainText()
    assert "Excluded <b>blurred-housing</b>.jpg" in dialog.output.toPlainText()
    assert service.failure in dialog.footer.text()
    assert len(warnings) == 1 and service.failure in warnings[0]
    dialog.close()


def test_evaluation_never_reads_training_screening_for_all_views(app):
    service = ScreeningService()
    service.evaluate = lambda view, backend, progress: {"scope": view, "parts": {"n": 2}}
    dialog = TrainingDialog(service, evaluate=True)
    assert dialog.view.currentData() == "all"
    dialog.start_job()
    wait_until(app, lambda: not dialog.busy)
    assert not any(call[0] == "screening" for call in service.calls)
    assert not dialog.screening_group.isVisible()
    dialog.close()


def test_provider_without_getter_can_show_embedded_training_report(app):
    class OlderService:
        config = {"product": "housing", "models": {"backend": "geometry"}}
        technician_unlocked = False
        def technician_configured(self):
            return True
        def dataset_stats(self):
            return {"images": 8}
        def train(self, view, backend, progress):
            return {"training_screening": screening_report()}
    dialog = TrainingDialog(OlderService())
    dialog.start_job()
    wait_until(app, lambda: not dialog.busy)
    assert "this training attempt" in dialog.screening_heading.text()
    assert "Accepted: 6 · Excluded: 2" in dialog.screening_output.toPlainText()
    dialog.close()
