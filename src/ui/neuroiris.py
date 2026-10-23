"""Responsive operator HMI for one-camera, multi-view part inspection."""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Callable

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSplitter, QTextEdit, QVBoxLayout, QWidget)

VIEW_LABELS = {"inner_face": "Face A · Inner machined", "ribbed_face": "Face B · Radial ribs",
               "side_left": "Left side", "side_right": "Right side"}
BACKEND_LABELS = {"geometry": "Geometry", "efficientad": "EfficientAD", "hybrid": "Hybrid"}
STYLE = """
QWidget {background:#000000;color:#FFFFFF;font-family:'Segoe UI',Arial;font-size:13px;}
QFrame#panel {background:#101512;border:1px solid #25342b;border-radius:9px;}
QLabel {background:transparent;}
QLabel#muted {color:#9aa99f;}
QPushButton {background:#14231b;color:#ecfff4;border:1px solid #305640;border-radius:6px;padding:11px 15px;font-weight:600;}
QPushButton:hover {background:#1b3c29;border-color:#00E676;}
QPushButton:pressed {background:#23603b;}
QPushButton:disabled {background:#101512;color:#58675e;border-color:#202c25;}
QPushButton#primary {background:#00E676;color:#001f0e;border:0;}
QPushButton#primary:hover {background:#58f5a0;}
QPushButton#primary:disabled {background:#123822;color:#638570;}
QPushButton#danger:disabled {color:#765b5b;border-color:#3b2929;}
QPushButton#danger {border-color:#843d3d;color:#ffaaa5;}
QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox {background:#131b16;border:1px solid #35483c;border-radius:5px;padding:8px;selection-background-color:#008a47;}
QComboBox QAbstractItemView {background:#142019;selection-background-color:#24663f;}
QTextEdit,QPlainTextEdit,QListWidget,QTableWidget {background:#0b100d;border:1px solid #2d3f33;border-radius:5px;gridline-color:#25342b;selection-background-color:#205b38;}
QHeaderView::section {background:#17251c;color:#bbd7c6;border:0;padding:8px;}
QGroupBox {border:1px solid #2b4133;border-radius:7px;margin-top:12px;padding-top:16px;}
QGroupBox::title {subcontrol-origin:margin;left:12px;color:#b6cabb;}
QProgressBar {border:1px solid #344c3d;border-radius:5px;text-align:center;background:#0b120e;min-height:17px;}
QProgressBar::chunk {background:#00a957;border-radius:4px;}
QTabWidget::pane {border:1px solid #263d2e;}
QTabBar::tab {background:#111c15;color:#bdd2c4;padding:10px 17px;}
QTabBar::tab:selected {background:#1d3a27;color:#00E676;}
QCheckBox::indicator {width:17px;height:17px;}
QScrollBar:vertical {background:#101812;width:10px;}
QScrollBar::handle:vertical {background:#35533f;min-height:25px;border-radius:4px;}
QToolTip {background:#203a29;color:white;border:1px solid #00E676;}
"""


def json_text(value: Any) -> str:
    def encode(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return {"shape": list(item.shape), "dtype": str(item.dtype)}
        if isinstance(item, np.generic):
            return item.item()
        if hasattr(item, "to_dict"):
            return item.to_dict()
        return str(item)
    return json.dumps(value, indent=2, default=encode, ensure_ascii=False)


class ImageCanvas(QLabel):
    """Keep the original pixels; scale only the rendered preview."""
    def __init__(self, text: str = "No image", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 200)
        self.setStyleSheet("background:#050806;border:1px solid #263c2d;border-radius:6px;color:#8caa97;")
        self._source: QPixmap | None = None
        self.image: np.ndarray | None = None

    def set_image(self, image: np.ndarray | None) -> None:
        if image is None:
            self.image = None
            self._source = None
            self.clear()
            self.setText("Camera preview unavailable")
            return
        self.image = image
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image)
        fmt = QImage.Format.Format_RGBA8888 if image.shape[2] == 4 else QImage.Format.Format_RGB888
        qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], fmt).copy()
        self._source = QPixmap.fromImage(qimage)
        self._render()

    def _render(self) -> None:
        if self._source is not None:
            self.setPixmap(self._source.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._render()


class TaskWorker(QObject):
    result = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    finished = pyqtSignal()

    def __init__(self, operation: Callable[[Callable[..., None]], Any]):
        super().__init__()
        self.operation = operation

    def report(self, *args: Any, **kwargs: Any) -> None:
        percent = 0
        messages = []
        for index, item in enumerate(args):
            if index == 0 and isinstance(item, (int, float)):
                percent = max(0, min(100, int(item)))
            else:
                messages.append(str(item))
        if kwargs:
            messages.append(json_text(kwargs))
        self.progress.emit(percent, " · ".join(messages))

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.result.emit(self.operation(self.report))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self, service: Any):
        super().__init__()
        self.service = service
        self._thread: QThread | None = None
        self._worker: TaskWorker | None = None
        self._last_sequence: Any = None
        self._latest: np.ndarray | None = None
        self.last_result: dict[str, Any] = {}
        self.session: dict[str, Any] | None = None
        self.camera_running = False
        self._inspection_started = 0.0
        self._job_started = 0.0
        self._job_name = ""
        self._dialogs: list[QWidget] = []
        self._model_readiness: dict[str, Any] | None = None
        self.setWindowTitle("NeuroIris — Aluminium Vision Inspection | Stavatech")
        self.setStyleSheet(STYLE)
        self._build()
        available = self.screen().availableGeometry() if self.screen() else None
        self.resize(min(1500, available.width()) if available else 1500,
                    min(980, available.height()) if available else 980)
        self.refresh_counts()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.poll_preview)
        self._timer.start(80)
        self._clock = QTimer(self)
        self._clock.timeout.connect(self.refresh_status)
        self._clock.start(1000)
        self.refresh_status()
        self.refresh_model_readiness()

    def _build(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 16)
        layout.setSpacing(14)
        header = QHBoxLayout()
        brand = QVBoxLayout()
        title = QLabel("Neuro<span style='color:#00E676'>Iris</span>")
        title.setStyleSheet("font-size:34px;font-weight:700;letter-spacing:1px;")
        brand.addWidget(title)
        subtitle = QLabel("ALUMINIUM VISION INSPECTION  ·  Stavatech Technology Private Limited")
        subtitle.setObjectName("muted")
        brand.addWidget(subtitle)
        header.addLayout(brand)
        header.addStretch()
        self.mode_label = QLabel("SINGLE CAMERA  /  TWO-FACE COVERAGE")
        self.mode_label.setStyleSheet("color:#00E676;font-size:12px;font-weight:600;padding:8px;")
        header.addWidget(self.mode_label)
        self.technician_access = QPushButton("Technician Access")
        self.technician_access.clicked.connect(self.toggle_technician)
        header.addWidget(self.technician_access)
        layout.addLayout(header)
        self.connection = QLabel("Camera stopped · actual resolution and FPS unavailable")
        self.connection.setObjectName("muted")
        layout.addWidget(self.connection)
        self.model_status = QLabel()
        self.model_status.setWordWrap(True)
        self.model_status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.model_status)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        live_panel = QFrame()
        live_panel.setObjectName("panel")
        live_layout = QVBoxLayout(live_panel)
        live_header = QHBoxLayout()
        self.canvas_label = QLabel("LIVE CAMERA")
        self.canvas_label.setStyleSheet("font-weight:600;color:#a5bdad;")
        live_header.addWidget(self.canvas_label)
        live_header.addStretch()
        self.show_live = QPushButton("Show Live Preview")
        self.show_live.clicked.connect(self.resume_preview)
        live_header.addWidget(self.show_live)
        live_layout.addLayout(live_header)
        self.canvas = ImageCanvas("Connect the camera to begin\n\nNative frames are preserved for analysis and evidence")
        live_layout.addWidget(self.canvas, 1)
        self.verdict = QLabel("READY")
        self.verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verdict.setStyleSheet("font-size:43px;font-weight:700;color:#a8b9ae;padding:10px;")
        live_layout.addWidget(self.verdict)
        self.reasons = QLabel("Start a part session, inspect both main faces, then finalize the part.")
        self.reasons.setWordWrap(True)
        self.reasons.setAlignment(Qt.AlignmentFlag.AlignCenter)
        live_layout.addWidget(self.reasons)
        splitter.addWidget(live_panel)
        side = QFrame()
        side.setObjectName("panel")
        side.setMinimumWidth(300)
        side.setMaximumWidth(440)
        right = QVBoxLayout(side)
        right.setContentsMargins(17, 18, 17, 15)
        guide = QLabel("PART INSPECTION")
        guide.setStyleSheet("font-size:17px;font-weight:650;color:#00E676;")
        right.addWidget(guide)
        note = QLabel("One physical housing · one final verdict\nPresent Face A and Face B in any order.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        right.addWidget(note)
        right.addWidget(QLabel("Physical part ID"))
        self.part_id = QLineEdit()
        self.part_id.setPlaceholderText("Scan or enter the housing ID")
        self.part_id.setMaxLength(100)
        right.addWidget(self.part_id)
        self.new_part = QPushButton("New Part Session")
        self.new_part.clicked.connect(self.begin_part)
        right.addWidget(self.new_part)
        right.addWidget(QLabel("Presented view"))
        self.view = QComboBox()
        self.view.addItem("Auto · identify presented face", "auto")
        for key, label in VIEW_LABELS.items():
            self.view.addItem(label, key)
        right.addWidget(self.view)
        self.coverage = QLabel("Required views: Face A + Face B\nNo part session active")
        self.coverage.setWordWrap(True)
        self.coverage.setStyleSheet("color:#b8ccbf;padding:10px 0;")
        right.addWidget(self.coverage)
        self.capture = QPushButton("Capture && Analyze View")
        self.capture.setObjectName("primary")
        self.capture.clicked.connect(self.capture_view)
        right.addWidget(self.capture)
        self.inspect_file = QPushButton("Inspect Saved Image…")
        self.inspect_file.clicked.connect(self.inspect_image)
        right.addWidget(self.inspect_file)
        self.finalize = QPushButton("Finalize Part")
        self.finalize.clicked.connect(self.finish_part)
        right.addWidget(self.finalize)
        self.cancel = QPushButton("Cancel Part Session")
        self.cancel.setObjectName("danger")
        self.cancel.clicked.connect(self.cancel_part)
        right.addWidget(self.cancel)
        right.addStretch()
        self.detail = QLabel("No inspection yet")
        self.detail.setWordWrap(True)
        self.detail.setObjectName("muted")
        right.addWidget(self.detail)
        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setMinimumWidth(320)
        side_scroll.setMaximumWidth(460)
        side_scroll.setWidget(side)
        splitter.addWidget(side_scroll)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        stats = QHBoxLayout()
        self.count_labels: dict[str, QLabel] = {}
        for label, key, color in [("TOTAL INSPECTED", "total", "#FFFFFF"), ("GOOD / PASS", "PASS", "#00E676"),
                                  ("NG / FAIL", "FAIL", "#ff5252"), ("RECHECK", "RECHECK", "#ffca54")]:
            panel = QFrame()
            panel.setObjectName("panel")
            vbox = QVBoxLayout(panel)
            caption = QLabel(label)
            caption.setObjectName("muted")
            value = QLabel("0")
            value.setStyleSheet(f"color:{color};font-size:26px;font-weight:650;")
            vbox.addWidget(caption)
            vbox.addWidget(value)
            stats.addWidget(panel)
            self.count_labels[key] = value
        layout.addLayout(stats)
        actions = QGridLayout()
        self.action_buttons: dict[str, QPushButton] = {}
        entries = [("START INSPECTION", self.start_inspection), ("STOP INSPECTION", self.stop_inspection),
                   ("TRAIN MODEL", self.train_model), ("EVALUATE MODEL", self.evaluate_model),
                   ("DATASET MANAGER", self.open_dataset), ("CAMERA SETTINGS", self.open_camera),
                   ("ROI EDITOR", self.open_roi), ("MODEL SETTINGS", self.open_models),
                   ("INSPECTION HISTORY", self.open_history), ("TECHNICAL SETTINGS", self.open_technical)]
        for index, (name, slot) in enumerate(entries):
            button = QPushButton(name)
            if index == 0:
                button.setObjectName("primary")
            button.clicked.connect(slot)
            actions.addWidget(button, index // 5, index % 5)
            self.action_buttons[name] = button
        layout.addLayout(actions)
        self.progress = QProgressBar()
        self.progress.hide()
        layout.addWidget(self.progress)
        self.status = QLabel("Ready · import separate physical parts before training.")
        self.status.setObjectName("muted")
        layout.addWidget(self.status)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(85)
        self.log.hide()
        layout.addWidget(self.log)
        self.setCentralWidget(root)
        self._showing_result = False
        self._update_actions()
        self._update_technician_label()

    def _update_actions(self) -> None:
        busy = self._thread is not None
        active = self.session is not None and str(self.session.get("status", "")).upper() not in {"COMPLETE", "FINALIZED", "CANCELLED", "RESULT_LOCKED"}
        capture_allowed = active and str(self.session.get("status", "")).upper() != "FAIL_LOCKED" if self.session else False
        self.new_part.setEnabled(not busy and not active)
        self.part_id.setEnabled(not active and not busy)
        self.capture.setEnabled(not busy and capture_allowed and self._latest is not None and self.camera_running)
        self.inspect_file.setEnabled(not busy and capture_allowed)
        self.finalize.setEnabled(not busy and active)
        self.cancel.setEnabled(not busy and active)
        self.view.setEnabled(not busy)
        for key, button in self.action_buttons.items():
            button.setEnabled(not busy)
        self.action_buttons["START INSPECTION"].setEnabled(not busy and not self.camera_running)
        self.action_buttons["STOP INSPECTION"].setEnabled(not busy and self.camera_running)

    def _error(self, message: str) -> None:
        self.status.setText(message)
        if self.verdict.text() == "ANALYZING":
            self._set_verdict("RECHECK")
            self.reasons.setText(message)
        self.log.show()
        self.log.append(message)
        QMessageBox.warning(self, "NeuroIris", message)

    def run_task(self, name: str, operation: Callable, done: Callable | None = None) -> None:
        if self._thread is not None:
            return
        self._job_name = name
        self._job_started = time.monotonic()
        self.status.setText(name)
        self.progress.setRange(0, 0)
        self.progress.show()
        thread = QThread(self)
        worker = TaskWorker(operation)
        worker.moveToThread(thread)
        self._thread, self._worker = thread, worker
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.result.connect(done or self._task_done)
        worker.failed.connect(self._error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._task_finished)
        thread.finished.connect(thread.deleteLater)
        self._update_actions()
        thread.start()

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(percent)
        self.status.setText(f"{self._job_name} · {message}")
        self.log.show()
        if message:
            self.log.append(message)

    def _task_done(self, result: Any) -> None:
        self.log.show()
        self.log.append(json_text(result))
        self.status.setText(f"{self._job_name} completed in {time.monotonic() - self._job_started:.1f}s")

    def _task_finished(self) -> None:
        self._thread = None
        self._worker = None
        self.progress.hide()
        self.refresh_counts()
        self.refresh_model_readiness()
        self._update_actions()

    def selected_view(self) -> str:
        return str(self.view.currentData())

    def begin_part(self) -> None:
        identity = self.part_id.text().strip()
        if not identity:
            self.part_id.setFocus()
            self._error("Enter a physical part ID before starting a session.")
            return
        try:
            self.session = self.service.begin_part(identity)
            self._inspection_started = time.monotonic()
            self._set_verdict("IN PROGRESS")
            self.reasons.setText("Capture the presented face, flip the housing, then capture the opposite face.")
            self._update_coverage()
            self.refresh_model_readiness()
            self._update_actions()
        except Exception as exc:
            self._error(str(exc))

    def _update_coverage(self) -> None:
        if not self.session:
            self.coverage.setText("No part session active")
            return
        missing = self.session.get("missing_views", [])
        required = self.service.config.get("views", {}).get("required", ["inner_face", "ribbed_face"])
        captured = self.session.get("views", {})
        if isinstance(captured, list):
            captured = {str(v): True for v in captured}
        lines = [f"Part: {self.session.get('part_id', self.part_id.text())}"]
        for view in required:
            label = VIEW_LABELS.get(view, view)
            state = "NEEDED" if view in missing else ("CAPTURED" if view in captured or not missing else "NEEDED")
            lines.append(f"{label}: {state}")
        if missing:
            lines.append("Flip or reposition the housing to capture missing views.")
        else:
            lines.append("Required captures complete. Finalize for one part verdict.")
        self.coverage.setText("\n".join(lines))

    def start_inspection(self) -> None:
        settings = self.service.config.get("camera", {})
        def open_camera(progress: Callable) -> None:
            self.service.start_camera(int(settings.get("index", 0)), int(settings.get("width", 3840)),
                                      int(settings.get("height", 2160)), float(settings.get("fps", 30)),
                                      str(settings.get("backend", "auto")))
        def opened(_: Any) -> None:
            self.camera_running = True
            self.resume_preview()
            self.status.setText("Acquisition started. Check the negotiated camera format before inspection.")
        self.run_task("Starting camera", open_camera, opened)

    def stop_inspection(self) -> None:
        def stopped(_: Any) -> None:
            self.camera_running = False
            self._latest = None
            if not self._showing_result:
                self.canvas.set_image(None)
            self.status.setText("Camera stopped. The current part session remains available.")
        self.run_task("Stopping camera", lambda progress: self.service.stop_camera(), stopped)

    def resume_preview(self) -> None:
        self._showing_result = False
        self.canvas_label.setText("LIVE CAMERA")
        self._last_sequence = None

    def poll_preview(self) -> None:
        if not self.camera_running:
            return
        try:
            packet = self.service.preview()
            if packet is None:
                self._latest = None
                self._last_sequence = None
                if not self._showing_result:
                    self.canvas.set_image(None)
                self._update_actions()
                return
            image = packet.get("image")
            sequence = packet.get("sequence")
            if isinstance(image, np.ndarray):
                self._latest = image
                if not self._showing_result and (sequence != self._last_sequence or sequence is None):
                    self.canvas.set_image(image)
                    self._last_sequence = sequence
            self._update_actions()
        except Exception as exc:
            self._latest = None
            self.connection.setText(f"Camera unavailable: {exc}")
            self._update_actions()

    def refresh_status(self) -> None:
        try:
            diagnostics = self.service.diagnostics()
            camera = diagnostics.get("camera", diagnostics.get("capture", {}))
            if not isinstance(camera, dict):
                camera = {}
            resolution = camera.get("actual_resolution")
            resolution_text = " × ".join(map(str, resolution)) if isinstance(resolution, (list, tuple)) else "unverified"
            fps = camera.get("actual_fps")
            fps_text = f"{fps:.1f} FPS" if isinstance(fps, (int, float)) else "FPS unavailable"
            status = camera.get("status", "stopped")
            backend = self.service.config.get("models", {}).get("backend", "geometry")
            self.connection.setText(f"Camera {status}  ·  {resolution_text}  ·  {fps_text}  ·  Model: {BACKEND_LABELS.get(backend, backend)}")
            self.connection.setStyleSheet("color:#00E676;" if camera.get("connected") else "color:#ffca54;")
            self.mode_label.setText("SINGLE CAMERA  /  " + ("EXTENDED COVERAGE" if len(self.service.config.get("views", {}).get("required", [])) > 2 else "TWO-FACE COVERAGE"))
        except Exception as exc:
            self.connection.setText(f"Diagnostics unavailable: {exc}")
        if self._thread is not None:
            self.status.setText(f"{self._job_name} · running for {time.monotonic() - self._job_started:.0f}s")

    def refresh_model_readiness(self, *_: Any) -> None:
        """Show saved-model availability without constructing inference models."""
        getter = getattr(self.service, "model_readiness", None)
        if not callable(getter):
            self._model_readiness = None
            self.model_status.hide()
            return
        try:
            report = getter()
            if not isinstance(report, dict):
                raise ValueError("Model availability was not reported")
        except Exception as exc:
            report = {"ready": False, "views": [], "message": f"Could not check saved models: {exc}"}
        self._model_readiness = report
        backend = report.get("backend", self.service.config.get("models", {}).get("backend", ""))
        summary = [f"Models · {BACKEND_LABELS.get(backend, backend)}"]
        details = [f"Saved models · {BACKEND_LABELS.get(backend, backend)}"]
        for row in report.get("views", []):
            label = VIEW_LABELS.get(row.get("view"), str(row.get("view", "Unknown view")))
            short_label = label.split(" · ", 1)[0]
            if row.get("ready"):
                summary.append(f"{short_label}: available")
                details.append(f"{label}: available · {row.get('version') or 'saved version'}")
            else:
                state = "missing" if row.get("status") == "missing" else "invalid"
                description = f"{short_label}: {state}"
                details.append(f"{label}: {row.get('reason') or 'No usable saved model'}")
                modes = row.get("available_modes", [])
                if modes:
                    description += " (" + ", ".join(BACKEND_LABELS.get(mode, mode) for mode in modes) + " saved)"
                summary.append(description)
        if not report.get("views") and report.get("message"):
            summary.append("availability could not be checked")
            details.append(str(report["message"]))
        lines = [" · ".join(summary)]
        if not report.get("ready") and report.get("artifact_root"):
            lines.append("Model folder: " + str(report["artifact_root"]))
            details.append(lines[-1])
        self.model_status.setText("\n".join(lines))
        self.model_status.setToolTip("\n".join(details))
        self.model_status.setStyleSheet("color:#b8ccbf;" if report.get("ready") else "color:#ffca54;")
        self.model_status.show()
        if not report.get("ready") and not self.last_result and self._thread is None:
            self._set_verdict("MODEL SETUP NEEDED")
            self.reasons.setText(str(report.get("message") or "Train or select a saved model for each required face before inspection."))
        elif report.get("ready") and self.verdict.text() == "MODEL SETUP NEEDED":
            self._set_verdict("IN PROGRESS" if self.session else "READY")
            self.reasons.setText("Capture the presented face, flip the housing, then capture the opposite face." if self.session else
                                 "Start a part session, inspect both main faces, then finalize the part.")

    def _capture_model_available(self, view: str) -> bool:
        self.refresh_model_readiness()
        report = self._model_readiness
        if report is None:
            return True
        rows = report.get("views", [])
        if view == "auto":
            ready = bool(report.get("ready"))
            blocked = [row for row in rows if not row.get("ready")]
        else:
            row = next((row for row in rows if row.get("view") == view), None)
            if row is None:
                try:
                    optional = self.service.model_readiness(views=[view])
                    row = next((row for row in optional.get("views", []) if row.get("view") == view), None)
                except Exception as exc:
                    row = {"view": view, "ready": False, "reason": str(exc)}
            ready = bool(row and row.get("ready"))
            blocked = [row] if row is not None and not ready else []
        if ready:
            return True
        reasons = [f"{VIEW_LABELS.get(row.get('view'), row.get('view'))}: {row.get('reason') or 'No usable saved model'}"
                   for row in blocked]
        message = "\n".join(reasons) or str(report.get("message") or f"No saved model is available for {VIEW_LABELS.get(view, view)}.")
        message += "\nOpen MODEL SETTINGS to select an available mode, or TRAIN MODEL for the required face and mode."
        self._set_verdict("MODEL SETUP NEEDED")
        self.reasons.setText(message)
        self.status.setText("Capture was not analyzed because its selected model is unavailable.")
        return False

    def capture_view(self) -> None:
        if self._latest is None:
            self._error("No current camera frame is available. Reconnect the selected camera.")
            return
        if not self.session:
            self._error("Start a part session first.")
            return
        view, identity = self.selected_view(), str(self.session.get("part_id", self.part_id.text()))
        if not self._capture_model_available(view):
            return
        self._set_verdict("ANALYZING")
        self.run_task("Selecting stable camera capture", lambda progress: self.service.inspect_live(view, identity), self.on_inspection)

    def inspect_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Inspect a saved native-resolution image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                self._error("The selected file is not a readable color image.")
            else:
                self.analyze(image)

    def analyze(self, image: np.ndarray) -> None:
        if not self.session:
            self._error("Start a part session first.")
            return
        view, identity = self.selected_view(), str(self.session.get("part_id", self.part_id.text()))
        if not self._capture_model_available(view):
            return
        self._set_verdict("ANALYZING")
        self.run_task("Analyzing captured view", lambda progress: self.service.inspect_frame(image, view, identity), self.on_inspection)

    def on_inspection(self, result: dict[str, Any]) -> None:
        self.last_result = result
        self.session = result.get("session", self.session)
        image = result.get("marked_image")
        if isinstance(image, np.ndarray):
            self.canvas.set_image(image)
            self._showing_result = True
        self.canvas_label.setText("CAPTURED EVIDENCE · " + VIEW_LABELS.get(result.get("view", ""), result.get("view", "unidentified view")))
        verdict = str(result.get("verdict", "RECHECK"))
        self._set_verdict(f"VIEW {verdict}")
        reasons = result.get("reasons", [])
        self.reasons.setText("\n".join(map(str, reasons)) if isinstance(reasons, list) else str(reasons))
        metrics = result.get("metrics", {})
        duration = metrics.get("total_ms", metrics.get("inference_ms")) if isinstance(metrics, dict) else None
        candidates = result.get("candidates", [])
        descriptions = [str(c.get("description", c.get("kind", c.get("source", "Candidate")))) if isinstance(c, dict) else str(c) for c in candidates]
        self.detail.setText(f"{len(candidates)} localized candidates" + (f" · {duration:.0f} ms" if isinstance(duration, (int, float)) else "") + ("\n" + "\n".join(descriptions[:5]) if descriptions else "") + "\nLast capture: " + datetime.now().strftime("%H:%M:%S"))
        self._update_coverage()
        self.status.setText("View analyzed. Review the evidence and capture any missing face before finalizing.")

    def _set_verdict(self, text: str) -> None:
        color = "#ff5252" if "FAIL" in text or "NG" == text else "#ffca54" if "RECHECK" in text or "SETUP NEEDED" in text else "#00E676" if "PASS" in text else "#a8b9ae"
        self.verdict.setText(text)
        self.verdict.setStyleSheet(f"font-size:40px;font-weight:700;color:{color};padding:8px;")

    def finish_part(self) -> None:
        def done(result: dict[str, Any]) -> None:
            self._set_verdict(str(result.get("verdict", "RECHECK")))
            reasons = result.get("reasons", [])
            self.reasons.setText("\n".join(map(str, reasons)) if isinstance(reasons, list) else str(reasons))
            self.session = None
            self.coverage.setText("Part finalized. Start a new part session for the next housing.")
            self.status.setText("One final part verdict saved to inspection history.")
            self.detail.setText(self.detail.text() + f"\nPart cycle: {time.monotonic() - self._inspection_started:.1f}s")
        self.run_task("Saving final part evidence", lambda progress: self.service.finish_part(), done)

    def cancel_part(self) -> None:
        def done(result: dict[str, Any]) -> None:
            self.session = None
            self.coverage.setText("Session cancelled. Start a new part session when ready.")
            self._set_verdict("CANCELLED")
            self.reasons.setText("Cancelled sessions do not increment finished-part counters.")
        self.run_task("Saving cancelled session", lambda progress: self.service.cancel_part(), done)

    def refresh_counts(self) -> None:
        try:
            rows = self.service.history()
            counts = {"total": 0, "PASS": 0, "FAIL": 0, "RECHECK": 0}
            for row in rows:
                if not row.get("counted", True) or not row.get("integrity_verified", False):
                    continue
                verdict = str(row.get("verdict", row.get("final_decision", {}).get("verdict", ""))).upper()
                verdict = {"GOOD": "PASS", "NG": "FAIL"}.get(verdict, verdict)
                if verdict in {"PASS", "FAIL", "RECHECK"}:
                    counts["total"] += 1
                    counts[verdict] += 1
            for key, label in self.count_labels.items():
                label.setText(str(counts[key]))
        except Exception as exc:
            self.status.setText(f"History unavailable: {exc}")

    def _open_dialog(self, dialog: QWidget) -> None:
        self._dialogs.append(dialog)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.destroyed.connect(lambda: self._dialogs.remove(dialog) if dialog in self._dialogs else None)
        dialog.destroyed.connect(self.refresh_model_readiness)
        if hasattr(dialog, "finished"):
            dialog.finished.connect(self.refresh_model_readiness)
        if hasattr(dialog, "models_changed"):
            dialog.models_changed.connect(self.refresh_model_readiness)
        dialog.show()

    def train_model(self) -> None:
        from src.ui.dialogs import TrainingDialog
        self._open_dialog(TrainingDialog(self.service, self, evaluate=False))

    def evaluate_model(self) -> None:
        from src.ui.dialogs import TrainingDialog
        self._open_dialog(TrainingDialog(self.service, self, evaluate=True))

    def open_dataset(self) -> None:
        from src.ui.dialogs import DatasetDialog
        self._open_dialog(DatasetDialog(self.service, self))

    def _update_technician_label(self) -> None:
        unlocked = bool(getattr(self.service, "technician_unlocked", False))
        self.technician_access.setText("Lock Technician" if unlocked else "Technician Access")

    def ensure_technician(self) -> bool:
        """Local commissioning guard; entered PINs never leave this process."""
        if not hasattr(self.service, "technician_configured"):
            return True
        if getattr(self.service, "technician_unlocked", False):
            return True
        try:
            if not self.service.technician_configured():
                pin, accepted = QInputDialog.getText(self, "Create commissioning PIN",
                    "Set a local technician PIN (at least 6 characters). This protects workstation settings; it is not enterprise authentication.",
                    QLineEdit.EchoMode.Password)
                if not accepted:
                    return False
                if len(pin) < 6:
                    self._error("The commissioning PIN must contain at least 6 characters.")
                    return False
                confirmation, accepted = QInputDialog.getText(self, "Confirm commissioning PIN",
                    "Enter the new technician PIN again.", QLineEdit.EchoMode.Password)
                if not accepted:
                    return False
                if pin != confirmation:
                    self._error("The commissioning PIN entries did not match.")
                    return False
                self.service.set_technician_pin(pin)
            else:
                pin, accepted = QInputDialog.getText(self, "Unlock technician settings",
                    "Enter the local commissioning PIN.", QLineEdit.EchoMode.Password)
                if not accepted:
                    return False
            if not self.service.unlock_technician(pin):
                self._error("The commissioning PIN was not accepted.")
                return False
            self._update_technician_label()
            return True
        except Exception as exc:
            self._error(str(exc))
            return False

    def toggle_technician(self) -> None:
        if hasattr(self.service, "lock_technician") and getattr(self.service, "technician_unlocked", False):
            self.service.lock_technician()
            self._update_technician_label()
            self.status.setText("Technician settings locked.")
        elif self.ensure_technician():
            self.status.setText("Technician settings unlocked for this application session.")

    def open_camera(self) -> None:
        if not self.ensure_technician():
            return
        from src.ui.dialogs import CameraDialog
        self._open_dialog(CameraDialog(self.service, self))

    def open_roi(self) -> None:
        if not self.ensure_technician():
            return
        from src.ui.roi_editor import ROIEditor
        view = self.selected_view()
        self._open_dialog(ROIEditor(self.service, view if view != "auto" else "inner_face", self._latest, self))

    def open_models(self) -> None:
        if not self.ensure_technician():
            return
        from src.ui.dialogs import ModelSettingsDialog
        self._open_dialog(ModelSettingsDialog(self.service, self))

    def open_history(self) -> None:
        from src.ui.dialogs import HistoryDialog
        self._open_dialog(HistoryDialog(self.service, self))

    def open_technical(self) -> None:
        if not self.ensure_technician():
            return
        from src.ui.dialogs import TechnicalDialog
        self._open_dialog(TechnicalDialog(self.service, self.last_result, self))

    def closeEvent(self, event: QCloseEvent) -> None:
        child_busy = any(getattr(dialog, "busy", False) for dialog in self._dialogs)
        if self._thread is not None or child_busy:
            QMessageBox.information(self, "Operation in progress", "Wait for the active training, evaluation, or capture operation to finish before closing.")
            event.ignore()
            return
        self._timer.stop()
        self._clock.stop()
        if hasattr(self.service, "lock_technician"):
            self.service.lock_technician()
        self.service.stop_camera()
        for dialog in list(self._dialogs):
            dialog.close()
        event.accept()
