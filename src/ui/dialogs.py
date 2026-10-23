"""Functional operator and technician dialogs for NeuroIris."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget)

from src.ui.neuroiris import BACKEND_LABELS, VIEW_LABELS, ImageCanvas, TaskWorker, json_text
from src.utils.filesystem import filesystem_path


def add_views(combo: QComboBox, selected: str = "inner_face") -> None:
    for key, label in VIEW_LABELS.items():
        combo.addItem(label, key)
    combo.setCurrentIndex(max(0, combo.findData(selected)))


def add_backends(combo: QComboBox, selected: str = "hybrid") -> None:
    for key, label in BACKEND_LABELS.items():
        combo.addItem(label, key)
    combo.setCurrentIndex(max(0, combo.findData(selected)))


def spin(value: int, minimum: int, maximum: int) -> QSpinBox:
    widget = QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    return widget


def decimal(value: float, minimum: float, maximum: float, decimals: int = 3) -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setDecimals(decimals)
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    return widget


def read_image(path: str) -> np.ndarray | None:
    try:
        # Python's Unicode/extended-path filesystem boundary avoids native
        # NumPy filename readers that still encounter Windows MAX_PATH limits.
        data = np.frombuffer(filesystem_path(path).read_bytes(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None


def error(parent: QWidget, message: Any) -> None:
    QMessageBox.warning(parent, "NeuroIris", str(message))


def export_json(parent: QWidget, value: Any, filename: str) -> None:
    path, _ = QFileDialog.getSaveFileName(parent, "Export report", filename, "JSON (*.json)")
    if path:
        try:
            filesystem_path(path).write_text(json_text(value), encoding="utf-8")
        except OSError as exc:
            error(parent, exc)


class WorkDialog(QDialog):
    """One retained worker thread, with explicit lifetime and real progress."""
    def __init__(self, service: Any, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.service = service
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(980, 740)
        self.busy = False
        self._thread: QThread | None = None
        self._worker: TaskWorker | None = None
        self._locked: list[QWidget] = []
        self.body = QVBoxLayout(self)
        self.footer = QLabel("")
        self.footer.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.hide()

    def finish_layout(self) -> None:
        self.body.addWidget(self.progress)
        self.body.addWidget(self.footer)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.body.addWidget(buttons)

    def run(self, operation: Callable, done: Callable | None = None) -> None:
        if self.busy:
            return
        self.busy = True
        for widget in self._locked:
            widget.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.footer.setText("Operation running…")
        thread = QThread(self)
        worker = TaskWorker(operation)
        worker.moveToThread(thread)
        self._thread, self._worker = thread, worker
        thread.started.connect(worker.run)
        worker.progress.connect(self._progress)
        worker.result.connect(done or (lambda result: self.footer.setText("Completed")))
        worker.failed.connect(self._failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _progress(self, percent: int, message: str) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(percent)
        self.footer.setText(message)

    def _failed(self, message: str) -> None:
        self.footer.setText(message)
        error(self, message)

    def _finished(self) -> None:
        self._thread = self._worker = None
        self.busy = False
        self.progress.hide()
        for widget in self._locked:
            widget.setEnabled(True)

    def reject(self) -> None:
        if self.busy:
            self.footer.setText("Wait for the current operation to finish before closing.")
            return
        super().reject()

    def closeEvent(self, event: Any) -> None:
        if self.busy:
            self.footer.setText("Wait for the current operation to finish before closing.")
            event.ignore()
        else:
            event.accept()


class DatasetDialog(WorkDialog):
    """Review managed samples and explicitly assign provenance to pending photos."""
    def __init__(self, service: Any, parent: QWidget | None = None):
        super().__init__(service, "Dataset Manager · physical-part provenance", parent)
        intro = QLabel("Assign the same physical part ID to every capture of one housing. Use one physical housing per import or repair batch; filenames never establish independent part IDs. If any capture of a housing is labelled NG, that whole housing is excluded from normal training, including its GOOD captures. Photos already in the dataset can be registered or repaired here.")
        intro.setWordWrap(True)
        self.body.addWidget(intro)
        fields = QGridLayout()
        self.view = QComboBox()
        add_views(self.view)
        self.label = QComboBox()
        self.label.addItems(["GOOD", "NG"])
        self.part_id = QLineEdit()
        self.part_id.setPlaceholderText("Individual housing ID, e.g. housing-017")
        self.part_id.setToolTip("Use one ID per actual physical housing, across both faces and every repeat capture. A product/model name shared by different housings is not a physical part ID.")
        self.session_id = QLineEdit()
        self.session_id.setPlaceholderText("Optional batch / capture session")
        for index, (label, widget) in enumerate([("View", self.view), ("Label", self.label), ("Physical part", self.part_id), ("Capture session", self.session_id)]):
            fields.addWidget(QLabel(label), 0, index)
            fields.addWidget(widget, 1, index)
        self.body.addLayout(fields)
        buttons = QGridLayout()
        self.import_button = QPushButton("Import Images…")
        self.import_button.clicked.connect(self.import_images)
        self.repair_button = QPushButton("Register / Repair Selected")
        self.repair_button.clicked.connect(self.repair_selected)
        self.capture_button = QPushButton("Capture Latest Camera Frame")
        self.capture_button.clicked.connect(self.capture_image)
        self.remove_button = QPushButton("Reject Selected Sample")
        self.remove_button.clicked.connect(self.remove_selected)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        for index, button in enumerate([self.import_button, self.repair_button, self.capture_button, self.remove_button, self.refresh_button]):
            buttons.addWidget(button, index // 3, index % 3)
        self.body.addLayout(buttons)
        self._locked = [self.import_button, self.repair_button, self.capture_button, self.remove_button,
                        self.refresh_button, self.view, self.label, self.part_id, self.session_id]
        split = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Status", "Part ID", "View", "Label", "Sample", "Capture session", "Details"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.review_selected)
        split.addWidget(self.table)
        self.preview = ImageCanvas("Select a saved or pending photo to review")
        split.addWidget(self.preview)
        self.body.addWidget(split, 1)
        self.statistics = QPlainTextEdit()
        self.statistics.setReadOnly(True)
        self.statistics.setMaximumHeight(110)
        self.body.addWidget(self.statistics)
        self.batch_report = QPlainTextEdit()
        self.batch_report.setReadOnly(True)
        self.batch_report.setMaximumHeight(145)
        self.batch_report.hide()
        self.body.addWidget(self.batch_report)
        self.finish_layout()
        self._samples: list[dict[str, Any]] = []
        self._pending: list[dict[str, Any]] = []
        self._entries: list[dict[str, Any]] = []
        self.refresh()

    def _identity(self) -> tuple[str, str, str, str] | None:
        identity = self.part_id.text().strip()
        if not identity:
            error(self, "A physical part ID is required. Enter the identity of the actual housing; do not create a different ID for each filename or capture.")
            return None
        return str(self.view.currentData()), self.label.currentText(), identity, self.session_id.text().strip()

    def _inventory(self) -> dict:
        samples = self.service.samples()
        statistics = self.service.dataset_stats()
        if hasattr(self.service, "pending_samples"):
            pending = self.service.pending_samples()
        else:
            # Keep older providers and lightweight UI test services usable.
            pending = statistics.get("pending", []) if isinstance(statistics, dict) else []
        return {"samples": samples, "statistics": statistics,
                "pending": pending if isinstance(pending, list) else []}

    def refresh(self) -> None:
        self.run(lambda progress: self._inventory(), self._apply_inventory)

    @staticmethod
    def _label_text(label: Any) -> str:
        return "GOOD" if label in (0, "GOOD", "good") else "NG" if label in (1, "NG", "ng") else str(label or "Unassigned")

    def _apply_inventory(self, inventory: dict) -> None:
        self._samples = inventory["samples"]
        self._pending = inventory.get("pending", [])
        self._entries = [{"pending": False, "sample": sample} for sample in self._samples]
        self._entries.extend({"pending": True, "sample": sample} for sample in self._pending)
        self.table.setRowCount(len(self._entries))
        for row, entry in enumerate(self._entries):
            sample = entry["sample"]
            metadata = sample.get("metadata") or {}
            pending = entry["pending"]
            needs_repair = sample.get("status") in {"invalid_metadata", "repair_required"}
            status = "Needs Repair" if pending and needs_repair else "Needs Registration" if pending else "Registered"
            view = sample.get("view")
            cells = [status, sample.get("part_id") or "Unassigned", VIEW_LABELS.get(view, view or "Select view above"),
                     self._label_text(sample.get("label")), Path(str(sample.get("path", ""))).name,
                     sample.get("session_id", metadata.get("session_id", "")), sample.get("reason", "")]
            for column, value in enumerate(cells):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(sample.get("path", "")) + ("\n" + str(sample["reason"]) if sample.get("reason") else ""))
                self.table.setItem(row, column, item)
        self.statistics.setPlainText(json_text(inventory["statistics"]))
        self.footer.setText(f"{len(self._samples)} registered samples · {len(self._pending)} photos need registration or repair." +
            (" Select pending photos of one housing, enter its physical part ID and view above, then click Register / Repair Selected." if self._pending else " Rejected samples are excluded from training."))
        self._update_selection_actions()

    def _selected_entries(self) -> list[dict]:
        selection = self.table.selectionModel()
        return [self._entries[index.row()] for index in sorted(selection.selectedRows(), key=lambda value: value.row())
                if 0 <= index.row() < len(self._entries)] if selection is not None else []

    def _update_selection_actions(self) -> None:
        selected = self._selected_entries()
        self.repair_button.setEnabled(not self.busy and bool(selected) and all(entry["pending"] for entry in selected)
                                      and hasattr(self.service, "repair_sample"))
        self.remove_button.setEnabled(not self.busy and len(selected) == 1 and not selected[0]["pending"])

    def _finished(self) -> None:
        super()._finished()
        self._update_selection_actions()

    def _run_batch(self, files: list[str], identity: tuple[str, str, str, str], *, repair: bool) -> None:
        operation_name = "Repair / registration" if repair else "Import"
        def operation(progress: Callable) -> dict:
            completed, failures = [], []
            action = self.service.repair_sample if repair else self.service.import_sample
            for index, path in enumerate(files):
                try:
                    completed.append({"source": path, "sample": action(path, *identity)})
                except Exception as exc:
                    failures.append({"source": path, "error": str(exc)})
                progress(int(100 * (index + 1) / len(files)), f"{operation_name}: {index + 1}/{len(files)} photos processed")
            # Preserve the outcome of every file even when a separate integrity
            # conflict prevents refreshing the rest of the inventory.
            result = {"completed": completed, "failures": failures, "identity": identity}
            try:
                result["inventory"] = self._inventory()
            except Exception as exc:
                result["inventory_error"] = str(exc)
            return result
        self.run(operation, self._batch_done)

    def _batch_done(self, result: dict) -> None:
        if "inventory" in result:
            self._apply_inventory(result["inventory"])
        completed, failures = result["completed"], result["failures"]
        view, label, part, session = result["identity"]
        lines = [f"{len(completed)} saved / registered; {len(failures)} failed.",
                 f"Requested physical part: {part} · View: {VIEW_LABELS.get(view, view)} · Label: {label}" + (f" · Session: {session}" if session else "")]
        for outcome in completed:
            sample = outcome["sample"]
            assigned = Path(str(sample.get("path", ""))).name
            lines.append(f"SAVED: {Path(outcome['source']).name} → {assigned}")
            stored_session = sample.get("session_id", (sample.get("metadata") or {}).get("session_id", ""))
            if str(stored_session) != session:
                lines.append(f"  Stored capture session: {stored_session or 'none'} (existing provenance retained).")
            warnings = sample.get("warnings", sample.get("warning"))
            if warnings:
                lines.append("  Warning: " + str(warnings))
        for outcome in failures:
            lines.append(f"FAILED: {Path(outcome['source']).name} — {outcome['error']}")
        if result.get("inventory_error"):
            lines.append("Inventory refresh blocked by a separate integrity problem: " + result["inventory_error"])
        self.batch_report.setPlainText("\n".join(lines))
        self.batch_report.show()
        self.footer.setText(f"{len(completed)} photos saved / registered · {len(failures)} failed. Review the named results above." +
                            (" Inventory refresh is blocked; resolve the reported integrity problem before training." if result.get("inventory_error") else ""))

    def import_images(self) -> None:
        identity = self._identity()
        if identity is None:
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Import captures of the selected physical part", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if files:
            self._run_batch(files, identity, repair=False)

    def repair_selected(self) -> None:
        selected = self._selected_entries()
        if not selected or not all(entry["pending"] for entry in selected):
            self.footer.setText("Select only photos marked Needs Registration or Needs Repair. Registered samples keep their existing identity.")
            return
        if not hasattr(self.service, "repair_sample"):
            error(self, "This dataset provider does not support metadata repair.")
            return
        identity = self._identity()
        if identity is not None:
            self._run_batch([str(entry["sample"]["path"]) for entry in selected], identity, repair=True)

    def capture_image(self) -> None:
        identity = self._identity()
        if identity is None:
            return
        try:
            packet = self.service.preview()
            if packet is None or not isinstance(packet.get("image"), np.ndarray):
                raise ValueError("No live camera frame. Start acquisition on the dashboard first.")
            image = packet["image"].copy()
        except Exception as exc:
            error(self, exc)
            return
        def operation(progress: Callable) -> dict:
            sample = self.service.capture_sample(image, *identity)
            result = {"completed": [{"source": "Live camera capture", "sample": sample}], "failures": [], "identity": identity}
            try:
                result["inventory"] = self._inventory()
            except Exception as exc:
                result["inventory_error"] = str(exc)
            return result
        self.run(operation, self._batch_done)

    def review_selected(self) -> None:
        self._update_selection_actions()
        index = self.table.currentRow()
        if 0 <= index < len(self._entries):
            self.preview.set_image(read_image(str(self._entries[index]["sample"].get("path", ""))))

    def remove_selected(self) -> None:
        selected = self._selected_entries()
        if len(selected) != 1 or selected[0]["pending"]:
            self.footer.setText("Only a registered sample can be rejected. Pending photos must be registered or repaired first.")
            return
        sample = selected[0]["sample"]
        def operation(progress: Callable) -> dict:
            self.service.remove_sample(str(sample["id"]))
            return self._inventory()
        def done(inventory: dict) -> None:
            self.preview.set_image(None)
            self._apply_inventory(inventory)
        self.run(operation, done)


class TrainingDialog(WorkDialog):
    models_changed = pyqtSignal()

    def __init__(self, service: Any, parent: QWidget | None = None, evaluate: bool = False):
        super().__init__(service, "Evaluate Model · blind held-out parts" if evaluate else "Train Model · view-specific learning", parent)
        self.evaluate = evaluate
        self.report: dict[str, Any] = {}
        self.screening_report: dict[str, Any] = {}
        self._screening_run_before: str | None = None
        self._screening_baseline_known = False
        self._screening_requested_view: str | None = None
        self._screening_attempt_active = False
        self._training_requested_view: str | None = None
        self._training_requested_backend: str | None = None
        self._training_can_activate = False
        self.footer.setTextFormat(Qt.TextFormat.PlainText)
        settings = service.config.get("models", {})
        form = QFormLayout()
        product = service.config.get("product", "Aluminium LED housing")
        self.product = QLabel(str(product.get("name", "Aluminium LED housing") if isinstance(product, dict) else product))
        form.addRow("Product", self.product)
        self.view = QComboBox()
        add_views(self.view)
        if evaluate:
            self.view.insertItem(0, "All required views · final part verdict", "all")
            self.view.setCurrentIndex(0)
        form.addRow("Housing face", self.view)
        self.backend = QComboBox()
        add_backends(self.backend, settings.get("backend", "hybrid"))
        form.addRow("Inspection backend", self.backend)
        neural = settings.get("efficientad", {})
        self.steps = spin(int(neural.get("max_steps", 10000)), 1, 1000000)
        self.batch = spin(int(neural.get("inference_batch_size", 2)), 1, 64)
        self.input_size = spin(int(neural.get("input_size", 512)), 64, 2048)
        self.imagenet = QLineEdit(str(neural.get("imagenet_dir") or "artifacts/pretrained/imagenette2/train"))
        self.teacher_cache = QLineEdit(str(neural.get("teacher_cache", "artifacts/pretrained/efficientad")))
        self.download_teacher = QCheckBox("Allow official checksum-verified teacher download (~40 MB)")
        self.download_teacher.setChecked(bool(neural.get("allow_download", False)))
        self.imagenette_status = QLabel("Imagenette downloads automatically when needed for training (~1.5 GB). Downloaded images are reused.")
        self.imagenette_status.setWordWrap(True)
        self.learning_rate = decimal(float(neural.get("learning_rate", .0001)), .000001, .1, 6)
        self.tile = spin(int(service.config.get("tiling", {}).get("tile_size", 512)), 64, 4096)
        self.overlap = decimal(float(service.config.get("tiling", {}).get("overlap", .25)), 0, .75, 2)
        ratios = service.config.get("dataset", {}).get("split", [.7, .15, .15])
        self.validation = decimal(float(ratios[1]), .05, .45, 2)
        self.test_fraction = decimal(float(ratios[2]), .05, .45, 2)
        self.group_sessions = QCheckBox("Keep each capture session / batch in one partition")
        self.group_sessions.setChecked(bool(service.config.get("dataset", {}).get("group_sessions", False)))
        if not evaluate:
            form.addRow("EfficientAD optimizer steps", self.steps)
            form.addRow("Training batch", QLabel("1 (required by EfficientAD)"))
            form.addRow("Neural inference batch", self.batch)
            form.addRow("Neural input edge (px)", self.input_size)
            form.addRow("Genuine ImageNet / Imagenette directory", self.imagenet)
            form.addRow("Teacher cache directory", self.teacher_cache)
            form.addRow(self.download_teacher)
            form.addRow(self.imagenette_status)
            form.addRow("Learning rate", self.learning_rate)
            form.addRow("Tile edge (native pixels)", self.tile)
            form.addRow("Tile overlap fraction", self.overlap)
            form.addRow("Validation part fraction", self.validation)
            form.addRow("Untouched test part fraction", self.test_fraction)
            form.addRow(self.group_sessions)
        setup = QWidget()
        setup.setLayout(form)
        setup_scroll = QScrollArea()
        setup_scroll.setWidgetResizable(True)
        setup_scroll.setWidget(setup)
        setup_scroll.setMinimumHeight(200)
        setup_scroll.setMaximumHeight(400)
        self.body.addWidget(setup_scroll)
        description = QLabel("Evaluation uses the same inference and decision pipeline on held-out physical parts. RECHECK remains in the report. No accuracy is reported before a real evaluation." if evaluate else "Training and reference selection use only training parts. Thresholds use validation parts. Untouched test parts stay separate. GOOD-only training is supported; sensitivity remains unvalidated without independent NG parts.")
        description.setWordWrap(True)
        self.body.addWidget(description)
        self.readiness = QLabel("")
        self.readiness.setWordWrap(True)
        self.readiness.setTextFormat(Qt.TextFormat.PlainText)
        self.readiness.setVisible(not evaluate)
        self.body.addWidget(self.readiness)
        self.stats = QPlainTextEdit()
        self.stats.setReadOnly(True)
        self.stats.setMaximumHeight(135)
        self.body.addWidget(self.stats)
        self.screening_group = QGroupBox("Automatic image screening · per training run")
        screening_layout = QVBoxLayout(self.screening_group)
        self.screening_heading = QLabel("")
        self.screening_heading.setTextFormat(Qt.TextFormat.PlainText)
        self.screening_heading.setWordWrap(True)
        screening_layout.addWidget(self.screening_heading)
        self.screening_output = QPlainTextEdit()
        self.screening_output.setReadOnly(True)
        self.screening_output.setMaximumHeight(160)
        screening_layout.addWidget(self.screening_output)
        self.screening_group.hide()
        self.body.addWidget(self.screening_group)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.body.addWidget(self.output, 1)
        row = QHBoxLayout()
        self.start = QPushButton("Evaluate Held-out Parts" if evaluate else "Train & Save Versioned Model")
        self.start.setObjectName("primary")
        self.start.clicked.connect(self.start_job)
        row.addWidget(self.start)
        self.benchmark_button = None
        if evaluate:
            self.benchmark_button = QPushButton("Benchmark Saved Models")
            self.benchmark_button.clicked.connect(self.start_benchmark)
            row.addWidget(self.benchmark_button)
            repeated = QPushButton("Repeated Sessions…")
            repeated.clicked.connect(lambda: RepeatabilityDialog(service, self).exec())
            row.addWidget(repeated)
        self.export = QPushButton("Export Report…")
        self.export.setEnabled(False)
        self.export.clicked.connect(lambda: export_json(self, self.report, "neuroiris-evaluation.json" if self.evaluate else "neuroiris-training.json"))
        row.addWidget(self.export)
        self.body.addLayout(row)
        self._locked = [self.start, self.view, self.backend, self.steps, self.batch, self.input_size, self.imagenet, self.teacher_cache, self.download_teacher, self.learning_rate,
                        self.tile, self.overlap, self.validation, self.test_fraction, self.group_sessions]
        if self.benchmark_button is not None:
            self._locked.append(self.benchmark_button)
        self._training_controls = [self.steps, self.batch, self.input_size, self.imagenet, self.teacher_cache,
            self.download_teacher, self.learning_rate, self.tile, self.overlap,
            self.validation, self.test_fraction, self.group_sessions]
        if not evaluate:
            self._training_controls.append(self.backend)
        self.finish_layout()
        self._apply_technician_restrictions()
        self.view.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        try:
            statistics = self.service.dataset_stats()
            readiness = statistics.get("training_readiness", {})
            # Show capture labels separately from whole-housing eligibility.
            # A saved GOOD image remains GOOD even when its housing has NG views.
            self.readiness.setText(str(readiness.get("message", "")))
            self.readiness.setStyleSheet("color: #f4ca64;" if readiness and not readiness.get("can_split") else "")
            self.stats.setPlainText(json_text(statistics))
        except Exception as exc:
            self.readiness.setText("")
            self.stats.setPlainText(str(exc))
        if not self.evaluate and not self.busy:
            self._refresh_screening(previous=True)

    def _append_output(self, text: str) -> None:
        # Filenames and quality errors are data, never Qt rich text.
        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(str(text) + "\n")
        self.output.setTextCursor(cursor)
        self.output.ensureCursorVisible()

    def _progress(self, percent: int, message: str) -> None:
        super()._progress(percent, message)
        if message:
            self._append_output(message)

    def start_job(self) -> None:
        if self.busy:
            return
        view, backend = str(self.view.currentData()), str(self.backend.currentData())
        allowed = self._can_edit_settings()
        if not self.evaluate and not allowed:
            # Operators use the commissioned mode, even if a programmatic UI
            # change tried to alter a disabled dropdown.
            backend = str(self.service.config.get("models", {}).get("backend", "hybrid"))
            self.backend.setCurrentIndex(max(0, self.backend.findData(backend)))
        if not self.evaluate and allowed:
            if self.validation.value() + self.test_fraction.value() >= .8:
                error(self, "Keep sufficient physical parts for training. Validation plus test fractions must be below 0.8.")
                return
            try:
                self.service.configure({"models": {"efficientad": {"max_steps": self.steps.value(), "train_batch_size": 1, "inference_batch_size": self.batch.value(), "input_size": self.input_size.value(), "learning_rate": self.learning_rate.value(), "imagenet_dir": self.imagenet.text().strip(), "teacher_cache": self.teacher_cache.text().strip(), "allow_download": self.download_teacher.isChecked()}},
                    "tiling": {"tile_size": self.tile.value(), "overlap": self.overlap.value()},
                    "dataset": {"split": [1 - self.validation.value() - self.test_fraction.value(), self.validation.value(), self.test_fraction.value()], "group_sessions": self.group_sessions.isChecked()}})
            except Exception as exc:
                error(self, exc)
                return
            self.refresh()
        self.output.clear()
        self.export.setEnabled(False)
        if not self.evaluate:
            self._training_requested_view = view
            self._training_requested_backend = backend
            self._training_can_activate = allowed
            self._begin_screening_attempt(view)
        operation = self.service.evaluate if self.evaluate else self.service.train
        self.run(lambda progress: operation(view, backend, progress), self.completed)

    def _read_screening(self, view: str) -> dict | None:
        getter = getattr(self.service, "training_screening", None)
        if self.evaluate or not callable(getter):
            return None
        report = getter(view)
        if not isinstance(report, dict) or not report:
            return None
        if report.get("view") not in (None, view):
            raise ValueError("The saved screening report belongs to another housing view")
        return report

    def _begin_screening_attempt(self, view: str) -> None:
        self._screening_run_before = None
        self._screening_baseline_known = False
        self._screening_requested_view = view
        self._screening_attempt_active = True
        self.screening_report = {}
        try:
            previous = self._read_screening(view)
            self._screening_baseline_known = callable(getattr(self.service, "training_screening", None))
            self._screening_run_before = str(previous["run_id"]) if previous and previous.get("run_id") else None
        except Exception:
            # A report read problem must not prevent a new training attempt.
            pass
        self.screening_output.clear()
        self.screening_heading.setText("This training attempt: waiting for image screening.")
        self.screening_group.setVisible(callable(getattr(self.service, "training_screening", None)))

    def _refresh_screening(self, report: dict | None = None, *, previous: bool = False) -> None:
        if self.evaluate:
            return
        provided_by_training_result = isinstance(report, dict) and bool(report)
        try:
            view = self._screening_requested_view if self._screening_attempt_active else str(self.view.currentData())
            if not isinstance(report, dict) or not report:
                report = self._read_screening(view)
            elif report.get("view") not in (None, view):
                raise ValueError("The training screening report belongs to another housing view")
        except Exception as exc:
            self.screening_group.show()
            self.screening_heading.setText("Training screening report could not be loaded.")
            self.screening_output.setPlainText(str(exc))
            return
        if not report:
            self.screening_report = {}
            if self._screening_attempt_active:
                self.screening_group.show()
                self.screening_heading.setText("No screening report is available for this training attempt.")
                self.screening_output.setPlainText("No per-file exclusion counts have been reported. Review the training error and progress log below.")
            else:
                self.screening_group.hide()
            return
        self.screening_report = report
        run_id = str(report.get("run_id", ""))
        if previous or not self._screening_attempt_active:
            scope = "Latest saved screening report · previous run"
        elif run_id and run_id != self._screening_run_before and (self._screening_baseline_known or provided_by_training_result):
            scope = "Screening report for this training attempt"
        elif run_id and run_id == self._screening_run_before:
            scope = "Previous run · this attempt has not produced a new screening report"
        else:
            scope = "Latest screening report · run identity could not be verified for this attempt"
        self.screening_heading.setText(scope)
        lines = [f"Run: {run_id or 'not reported'} · Status: {report.get('status', 'not reported')}",
                 f"View: {report.get('view', view)} · Backend: {report.get('backend', 'not reported')}",
                 f"Candidate images: {report.get('candidate_images', 'not reported')} · Accepted: {report.get('accepted_images', 'not reported')} · Excluded: {report.get('excluded_images', 'not reported')}",
                 "Original files and labels are retained. Exclusions apply only to this training run."]
        validation = report.get("validation_screening", {})
        if isinstance(validation, dict) and validation:
            lines.append(f"Independent GOOD validation: {validation.get('good_accepted_images', 0)} of {validation.get('good_candidate_images', 0)} accepted · Status: {validation.get('status', 'not reported')}")
            lines.append("Reserved GOOD physical parts: " + ", ".join(map(str, validation.get("reserved_good_part_ids", []))))
            if validation.get("expected_source_resolutions"):
                lines.append("Training source resolutions (width × height): " + ", ".join(
                    f"{width} × {height}" for width, height in validation["expected_source_resolutions"]))
        invalid_validation = report.get("invalid_validation_captures", [])
        if isinstance(invalid_validation, list) and invalid_validation:
            lines.append("Validation captures requiring recapture (kept in their reserved split):")
            for entry in invalid_validation:
                if not isinstance(entry, dict):
                    continue
                filename = entry.get("filename") or entry.get("path") or entry.get("sample_id") or "Unnamed sample"
                lines.append(f"- {filename} · Part: {entry.get('part_id', 'not reported')} · Stage: {entry.get('stage', 'analysis')}")
                if entry.get("path"):
                    lines.append("  File: " + str(entry["path"]))
                reasons = entry.get("reasons", [])
                lines.append("  Reasons: " + ("; ".join(map(str, reasons)) if isinstance(reasons, list) else str(reasons)))
                if entry.get("metrics"):
                    lines.append("  Measured quality: " + json_text(entry["metrics"]))
        excluded = report.get("excluded", [])
        if isinstance(excluded, list) and excluded:
            lines.append("Excluded files:")
            for entry in excluded:
                if not isinstance(entry, dict):
                    continue
                filename = entry.get("filename") or entry.get("path") or entry.get("sample_id") or "Unnamed sample"
                lines.append(f"- {filename} · Part: {entry.get('part_id', 'not reported')} · Stage: {entry.get('stage', 'not reported')}")
                if entry.get("path"):
                    lines.append("  File: " + str(entry["path"]))
                reasons = entry.get("reasons", [])
                lines.append("  Reasons: " + ("; ".join(map(str, reasons)) if isinstance(reasons, list) else str(reasons)))
                if entry.get("metrics"):
                    lines.append("  Measured quality: " + json_text(entry["metrics"]))
        if report.get("error"):
            lines.append("Run error: " + str(report["error"]))
        if report.get("report_path"):
            lines.append("Report path: " + str(report["report_path"]))
        lines.append("Review excluded photos and recapture unfit images before a later training run.")
        self.screening_output.setPlainText("\n".join(lines))
        self.screening_group.show()

    def _failed(self, message: str) -> None:
        if not self.evaluate:
            self._refresh_screening()
            self._append_output("Training failed: " + message)
        super()._failed(message)

    def _can_edit_settings(self) -> bool:
        return (not hasattr(self.service, "technician_configured") or
                bool(getattr(self.service, "technician_unlocked", False)))

    def _apply_technician_restrictions(self) -> None:
        allowed = self._can_edit_settings()
        for control in self._training_controls:
            control.setEnabled(allowed and not self.busy)
        if not allowed and not self.evaluate and not self.footer.text():
            self.footer.setText("Operator mode: training uses saved settings. Unlock Technician Access on the dashboard to change commissioning settings.")

    def _finished(self) -> None:
        super()._finished()
        self._screening_attempt_active = False
        self._apply_technician_restrictions()

    def start_benchmark(self) -> None:
        if not hasattr(self.service, "benchmark"):
            error(self, "This service has no saved-model benchmark adapter.")
            return
        self.output.clear()
        self.export.setEnabled(False)
        view = str(self.view.currentData())
        self.run(lambda progress: self.service.benchmark(view, progress), self.completed)

    def completed(self, report: dict) -> None:
        self.report = report
        self._append_output(json_text(report))
        self.export.setEnabled(True)
        if self.evaluate:
            self.footer.setText("Evaluation completed. Review sample counts, coverage and false rejects before selecting this model.")
            return
        self._refresh_screening(report.get("training_screening"))
        view = self._training_requested_view or str(report.get("view") or self.view.currentData())
        requested = self._training_requested_backend or str(self.backend.currentData())
        trained = str(report.get("backend") or ("efficientad" if requested == "hybrid" else requested))
        active = str(self.service.config.get("models", {}).get("backend", "hybrid"))
        warning = ""
        expected = "efficientad" if requested == "hybrid" else requested
        if trained != expected:
            warning = "The saved backend differs from the requested mode. Select the compatible saved model in Model Settings."
        elif requested != active:
            if self._training_can_activate and self._can_edit_settings():
                try:
                    select_mode = getattr(self.service, "set_mode", None)
                    if callable(select_mode):
                        select_mode(requested)
                    else:
                        self.service.configure({"models": {"backend": requested}})
                except Exception as exc:
                    # The model is already published. Keep its successful report
                    # and export available even if a settings write fails.
                    warning = f"Model saved, but activating {BACKEND_LABELS.get(requested, requested)} failed: {exc}. Select it in Model Settings."
            else:
                warning = "Model saved. Unlock Technician Access and select its mode in Model Settings before inspection."
            active = str(self.service.config.get("models", {}).get("backend", "hybrid"))
            if not warning and active != requested:
                warning = "Model saved, but the active mode did not change. Select its mode in Model Settings before inspection."
        lines = [f"Training completed for {VIEW_LABELS.get(view, view)}. Trained backend: {BACKEND_LABELS.get(trained, trained)}.",
                 f"Active inspection mode: {BACKEND_LABELS.get(active, active)}."]
        if report.get("artifact_path"):
            lines.append("Saved model folder: " + str(report["artifact_path"]))
        if warning:
            lines.append(warning)
        lines.append("Review independent validation before using this model.")
        self.footer.setText("\n".join(lines))
        self.models_changed.emit()


class CameraDialog(WorkDialog):
    def __init__(self, service: Any, parent: QWidget | None = None):
        super().__init__(service, "Camera Settings · selected device only", parent)
        self.resize(800, 700)
        settings = service.config.get("camera", {})
        form = QFormLayout()
        self.index = spin(int(settings.get("index", 0)), 0, 64)
        self.backend = QComboBox()
        for name in ["auto", "dshow", "msmf", "v4l2", "any"]:
            self.backend.addItem(name, name)
        self.backend.setCurrentIndex(max(0, self.backend.findData(settings.get("backend", "auto"))))
        self.width = spin(int(settings.get("width", 3840)), 160, 16384)
        self.height = spin(int(settings.get("height", 2160)), 120, 16384)
        self.fps = decimal(float(settings.get("fps", 30)), 1, 240, 1)
        for name, widget in [("Camera index", self.index), ("Capture backend", self.backend), ("Requested width (px)", self.width), ("Requested height (px)", self.height), ("Requested FPS", self.fps)]:
            form.addRow(name, widget)
        self.body.addLayout(form)
        note = QLabel("MJPEG is requested before 4K negotiation. Requested settings are preferences; the live diagnostics below show the actual format. Device index never falls back to a different camera.")
        note.setWordWrap(True)
        self.body.addWidget(note)
        self.apply = QPushButton("Save & Reconnect Selected Camera")
        self.apply.clicked.connect(self.connect_camera)
        self.body.addWidget(self.apply)
        controls = QGroupBox("Driver controls · raw backend units")
        control_layout = QFormLayout(controls)
        self._unsupported_controls: set[str] = set()
        self.control = QComboBox()
        for name in ["exposure", "auto_exposure", "gain", "brightness", "focus", "autofocus", "white_balance", "auto_white_balance"]:
            self.control.addItem(name.replace("_", " ").title(), name)
        self.value = decimal(0, -100000, 100000, 3)
        self.apply_control = QPushButton("Apply & Verify Readback")
        self.apply_control.clicked.connect(self.set_control)
        control_layout.addRow("Control", self.control)
        control_layout.addRow("Raw device value", self.value)
        control_layout.addRow(self.apply_control)
        self.control_note = QLabel("Control support is confirmed only after the driver accepts a value and verifies readback. Auto-exposure encodings depend on the backend. Use the device documentation.")
        self.control_note.setWordWrap(True)
        control_layout.addRow(self.control_note)
        self.body.addWidget(controls)
        self.control.currentIndexChanged.connect(lambda: self.refresh())
        self.diagnostics = QPlainTextEdit()
        self.diagnostics.setReadOnly(True)
        self.body.addWidget(self.diagnostics, 1)
        self._locked = [self.apply, self.index, self.backend, self.width, self.height, self.fps, self.apply_control]
        self.finish_layout()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(1000)
        self.refresh()

    def refresh(self) -> None:
        try:
            data = self.service.diagnostics()
            camera = data.get("camera", data.get("capture", {}))
            self.diagnostics.setPlainText(json_text(camera))
            if not self.busy:
                supported = hasattr(self.service, "set_camera_control")
                available = supported and bool(camera.get("connected")) and self.control.currentData() not in self._unsupported_controls
                self.apply_control.setEnabled(available)
                self.value.setEnabled(available)
                if not supported:
                    self.control_note.setText("Camera driver control API is unavailable in this provider.")
        except Exception as exc:
            self.diagnostics.setPlainText(str(exc))

    def _finished(self) -> None:
        super()._finished()
        self.refresh()

    def connect_camera(self) -> None:
        settings = {"index": self.index.value(), "backend": self.backend.currentData(),
                    "width": self.width.value(), "height": self.height.value(), "fps": self.fps.value()}
        def operation(progress: Callable) -> None:
            self.service.configure({"camera": settings})
            self._unsupported_controls.clear()
            self.service.stop_camera()
            self.service.start_camera(**settings)
        def done(_: Any) -> None:
            parent = self.parent()
            if hasattr(parent, "camera_running"):
                parent.camera_running = True
                parent.resume_preview()
            self.footer.setText("Acquisition started. Verify actual resolution, MJPEG format and throughput in diagnostics.")
            self.refresh()
        self.run(operation, done)

    def set_control(self) -> None:
        name, value = str(self.control.currentData()), self.value.value()
        def done(verified: bool) -> None:
            self.control_note.setText("Driver accepted the control and verified readback." if verified else "Control unavailable for this device/backend/value: driver rejected it or readback did not match. Inspect diagnostics and device documentation.")
            if not verified:
                self._unsupported_controls.add(name)
                self.apply_control.setEnabled(False)
                self.value.setEnabled(False)
            self.refresh()
        self.run(lambda progress: self.service.set_camera_control(name, value), done)


class ModelSettingsDialog(QDialog):
    def __init__(self, service: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self.service = service
        self.setWindowTitle("Model Settings · saved view-specific versions")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(900, 640)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.backend = QComboBox()
        add_backends(self.backend, service.config.get("models", {}).get("backend", "hybrid"))
        form.addRow("Active inference mode", self.backend)
        self.view = QComboBox()
        add_views(self.view)
        form.addRow("Model view", self.view)
        self.models = QComboBox()
        form.addRow("Saved model version", self.models)
        layout.addLayout(form)
        self.metadata = QPlainTextEdit()
        self.metadata.setReadOnly(True)
        layout.addWidget(self.metadata, 1)
        optional = QLabel("Dinomaly benchmark and DINOv2 comparison: unavailable in this release. They are disabled and never substituted for trained neural models.")
        optional.setWordWrap(True)
        layout.addWidget(optional)
        self.side_views = QCheckBox("Require both side views as well as Face A and Face B")
        self.side_views.setChecked(len(service.config.get("views", {}).get("required", [])) > 2)
        layout.addWidget(self.side_views)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._models: list[dict] = []
        self.view.currentIndexChanged.connect(self.refresh)
        self.backend.currentIndexChanged.connect(self.refresh)
        self.models.currentIndexChanged.connect(self.show_metadata)
        self.refresh()

    def refresh(self) -> None:
        try:
            self._models = [model for model in self.service.models() if model.get("view") == self.view.currentData() and model.get("backend") == ("efficientad" if self.backend.currentData() == "hybrid" else self.backend.currentData())]
            self.models.clear()
            self.models.addItem("Use currently selected model", None)
            for model in self._models:
                self.models.addItem(str(model.get("version", model.get("id", "Saved model"))), model)
            self.show_metadata()
        except Exception as exc:
            self.metadata.setPlainText(str(exc))

    def show_metadata(self) -> None:
        selected = self.models.currentData()
        self.metadata.setPlainText(json_text(selected) if selected else "Saved model selection is resolved per view. A missing or unvalidated model produces RECHECK; selecting a mode does not fabricate a trained model.")

    def save(self) -> None:
        try:
            backend = str(self.backend.currentData())
            self.service.set_mode(backend)
            required = ["inner_face", "ribbed_face"] + (["side_left", "side_right"] if self.side_views.isChecked() else [])
            self.service.configure({"views": {"required": required, "enabled": list(VIEW_LABELS)}})
            model = self.models.currentData()
            if model:
                self.service.select_model(str(self.view.currentData()), str(model["version"]), backend)
            self.accept()
        except Exception as exc:
            error(self, exc)


class HistoryDialog(QDialog):
    @staticmethod
    def display_verdict(record: dict) -> str:
        decision = record.get("final_decision") or {}
        verdict = str(record.get("verdict", decision.get("verdict", "")))
        if record.get("inspection_mode") == "manual":
            return {"PASS": "OK", "FAIL": "NG", "RECHECK": "View Recheck"}.get(verdict, verdict)
        return verdict

    def __init__(self, service: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self.service = service
        self.rows: list[dict] = []
        self.setWindowTitle("Inspection History · one event per finalized part")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(1000, 700)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Physical part", "Verdict", "Time", "Views", "Evidence"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.selected)
        layout.addWidget(self.table, 1)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(230)
        layout.addWidget(self.details)
        actions = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        export = QPushButton("Export History CSV…")
        export.clicked.connect(self.export_csv)
        export_record = QPushButton("Export Selected Record JSON…")
        export_record.clicked.connect(self.export_record)
        repeated = QPushButton("Repeatability Summary…")
        repeated.clicked.connect(lambda: RepeatabilityDialog(service, self).exec())
        for button in [refresh, export, export_record, repeated]:
            actions.addWidget(button)
        layout.addLayout(actions)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh()

    def refresh(self) -> None:
        try:
            self.rows = list(reversed(self.service.history()))
            self.table.setRowCount(len(self.rows))
            for row, record in enumerate(self.rows):
                values = [record.get("part_id", ""), self.display_verdict(record), record.get("timestamp", record.get("finished_at", "")),
                          ", ".join(record.get("views", {}).keys()) if isinstance(record.get("views"), dict) else str(record.get("views", "")),
                          record.get("evidence_path", record.get("path", ""))]
                for column, value in enumerate(values):
                    self.table.setItem(row, column, QTableWidgetItem(str(value)))
        except Exception as exc:
            error(self, exc)

    def selected(self) -> None:
        index = self.table.currentRow()
        self.details.setPlainText(json_text(self.rows[index]) if 0 <= index < len(self.rows) else "")

    def export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export finalized part history", "neuroiris-history.csv", "CSV (*.csv)")
        if path:
            try:
                with filesystem_path(path).open("w", newline="", encoding="utf-8-sig") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["part_id", "verdict", "timestamp", "reasons", "missing_views", "record"])
                    writer.writeheader()
                    for row in self.rows:
                        writer.writerow({"part_id": row.get("part_id"), "verdict": self.display_verdict(row), "timestamp": row.get("timestamp", row.get("finished_at", "")),
                                         "reasons": json_text(row.get("reasons", [])), "missing_views": json_text(row.get("missing_views", [])), "record": json_text(row)})
            except OSError as exc:
                error(self, exc)

    def export_record(self) -> None:
        index = self.table.currentRow()
        if 0 <= index < len(self.rows):
            export_json(self, self.rows[index], "neuroiris-inspection.json")


def repeatability_summary(records: list[dict]) -> dict:
    """Summarize verified repeated sessions without treating consistency as accuracy."""
    grouped: dict[str, list[dict]] = {}
    seen = set()
    for record in records:
        if not record.get("counted") or not record.get("integrity_verified"):
            continue
        event = record.get("event_id")
        if event and event in seen:
            continue
        if event:
            seen.add(event)
        part = str(record.get("part_id", "")).strip()
        if part:
            grouped.setdefault(part, []).append(record)
    groups = []
    for part, sessions in sorted(grouped.items()):
        if len(sessions) < 2:
            continue
        counts = {"PASS": 0, "FAIL": 0, "RECHECK": 0}
        versions = set()
        required = set()
        for session in sessions:
            verdict = session.get("verdict", (session.get("final_decision") or {}).get("verdict", "RECHECK"))
            if verdict in counts:
                counts[verdict] += 1
            versions.add(tuple(sorted((view, str(decision.get("model_version") or "unrecorded"))
                                     for view, decision in session.get("views", {}).items())))
            required.add(tuple(sorted(session.get("required_views", []))))
        groups.append({"part_id": part, "sessions": len(sessions), "verdict_counts": counts,
                       "consistent_verdict": sum(value > 0 for value in counts.values()) == 1,
                       "same_recorded_models_and_coverage": (len(versions) == 1 and len(required) == 1
                           and all(signature and all(version != "unrecorded" for _, version in signature) for signature in versions)),
                       "model_signatures": [list(signature) for signature in sorted(versions)],
                       "event_ids": [session.get("event_id") for session in sessions]})
    return {"scope": "Recorded repeated inspection sessions of the same physical part ID",
            "limitations": ["Operator must use the same physical housing and reposition it between sessions.",
                            "History does not verify physical repositioning or defect ground truth.",
                            "Consistent decisions are not a measure of defect detection accuracy.",
                            "Model and coverage changes must be reviewed before comparing sessions."],
            "independent_part_ids": len(grouped), "repeated_part_ids": len(groups),
            "repeated_sessions": sum(group["sessions"] for group in groups),
            "parts_with_mixed_verdicts": sum(not group["consistent_verdict"] for group in groups),
            "groups": groups}


class RepeatabilityDialog(WorkDialog):
    def __init__(self, service: Any, parent: QWidget | None = None):
        super().__init__(service, "Repeatability · recorded physical-part sessions", parent)
        self.report: dict = {}
        instruction = QLabel("To collect repeated placements, finalize the first inspection, physically remove and reposition the same housing, then start a new session with the same part ID. Capture every required view again. Use the same models and settings. This report counts verified stored sessions; it does not measure detection accuracy.")
        instruction.setWordWrap(True)
        self.body.addWidget(instruction)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Physical part", "Sessions", "PASS", "FAIL", "RECHECK", "Verdicts", "Model / coverage"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.body.addWidget(self.table, 1)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(220)
        self.body.addWidget(self.details)
        actions = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh Verified Sessions")
        self.refresh_button.clicked.connect(self.refresh)
        actions.addWidget(self.refresh_button)
        self.export_button = QPushButton("Export Summary JSON…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(lambda: export_json(self, self.report, "neuroiris-repeatability.json"))
        actions.addWidget(self.export_button)
        self.body.addLayout(actions)
        self._locked = [self.refresh_button]
        self.finish_layout()
        self.refresh()

    def refresh(self) -> None:
        self.run(lambda progress: repeatability_summary(self.service.history()), self.completed)

    def completed(self, report: dict) -> None:
        self.report = report
        self.table.setRowCount(len(report["groups"]))
        for row, group in enumerate(report["groups"]):
            counts = group["verdict_counts"]
            values = [group["part_id"], group["sessions"], counts["PASS"], counts["FAIL"], counts["RECHECK"],
                      "Consistent" if group["consistent_verdict"] else "MIXED · review",
                      "Same recorded models" if group["same_recorded_models_and_coverage"] else "Changed / incomplete"]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))
        self.details.setPlainText(json_text(report))
        self.export_button.setEnabled(True)
        self.footer.setText(f"{report['repeated_part_ids']} repeated physical part IDs · {report['repeated_sessions']} finalized sessions · {report['parts_with_mixed_verdicts']} parts with mixed verdicts")


class TechnicalDialog(QDialog):
    def __init__(self, service: Any, last_result: dict, parent: QWidget | None = None):
        super().__init__(parent)
        self.service = service
        self.last_result = last_result
        if hasattr(service, "diagnostic_maps"):
            last_result = {**last_result, "diagnostic_maps": service.diagnostic_maps()}
        self.setWindowTitle("Technician · diagnostics and image-unit tolerances")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(1050, 800)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        self.diagnostics = QPlainTextEdit()
        self.diagnostics.setReadOnly(True)
        tabs.addTab(self.diagnostics, "Hardware & camera")
        self.result = QPlainTextEdit()
        self.result.setReadOnly(True)
        self.result.setPlainText(json_text(last_result))
        tabs.addTab(self.result, "Last inference details")
        maps = QWidget()
        map_layout = QVBoxLayout(maps)
        self.map_choice = QComboBox()
        self.map_images: dict[str, np.ndarray] = {}
        self._find_maps(last_result)
        for name in self.map_images:
            self.map_choice.addItem(name)
        map_layout.addWidget(self.map_choice)
        self.map_canvas = ImageCanvas("No diagnostic maps available. Inspect a view first.")
        map_layout.addWidget(self.map_canvas, 1)
        self.map_choice.currentTextChanged.connect(self.display_map)
        tabs.addTab(maps, "Maps & registration")
        settings_widget = QWidget()
        form = QFormLayout(settings_widget)
        note = QLabel("Changes apply to subsequent training or inspection. Pixel tolerances are image measurements; no micrometre precision is claimed. Recalibrate and validate thresholds after changing the pipeline.")
        note.setWordWrap(True)
        form.addRow(note)
        self.settings = QPlainTextEdit()
        editable = {key: service.config.get(key, {}) for key in ["quality", "geometry", "calibration", "tiling"]}
        self.settings.setPlainText(json_text(editable))
        form.addRow("Advanced configuration (JSON)", self.settings)
        self.apply = QPushButton("Validate JSON & Save Overrides")
        self.apply.clicked.connect(self.save)
        form.addRow(self.apply)
        tabs.addTab(settings_widget, "Tolerances & calibration")
        layout.addWidget(tabs, 1)
        row = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh Diagnostics")
        self.refresh_button.clicked.connect(self.refresh)
        row.addWidget(self.refresh_button)
        export = QPushButton("Export Diagnostic Report…")
        export.clicked.connect(lambda: export_json(self, {"diagnostics": service.diagnostics(), "last_result": last_result}, "neuroiris-diagnostics.json"))
        row.addWidget(export)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(2000)
        self.refresh()
        self.display_map(self.map_choice.currentText())

    def _find_maps(self, data: Any, prefix: str = "") -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, np.ndarray) and value.ndim in (2, 3) and value.size:
                    self.map_images[path] = value
                elif isinstance(value, dict):
                    self._find_maps(value, path)

    def display_map(self, name: str) -> None:
        image = self.map_images.get(name)
        if image is None:
            return
        if image.dtype != np.uint8:
            finite = np.nan_to_num(image)
            if finite.size and float(finite.max()) > float(finite.min()):
                image = np.clip((finite - finite.min()) / (finite.max() - finite.min()) * 255, 0, 255).astype(np.uint8)
            else:
                image = np.zeros_like(finite, dtype=np.uint8)
        if image.ndim == 2:
            image = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
        self.map_canvas.set_image(image)

    def refresh(self) -> None:
        try:
            self.diagnostics.setPlainText(json_text(self.service.diagnostics()))
        except Exception as exc:
            self.diagnostics.setPlainText(str(exc))

    def save(self) -> None:
        try:
            settings = json.loads(self.settings.toPlainText())
            if not isinstance(settings, dict) or any(key not in {"quality", "geometry", "calibration", "tiling"} for key in settings):
                raise ValueError("Only quality, geometry, calibration and tiling objects may be edited here.")
            if any(not isinstance(value, dict) for value in settings.values()):
                raise ValueError("Each configuration section must be a JSON object.")
            self.service.configure(settings)
            QMessageBox.information(self, "Configuration saved", "Overrides saved. Validate the affected pipeline before relying on its decisions.")
        except Exception as exc:
            error(self, exc)
