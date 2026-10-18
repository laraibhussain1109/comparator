"""Native-image ROI drawing and vertex editing with normalized persistence."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget)

from src.ui.neuroiris import VIEW_LABELS


class ROICanvas(QWidget):
    region_created = pyqtSignal(dict)
    regions_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.image: np.ndarray | None = None
        self.pixmap: QPixmap | None = None
        self.regions: list[dict] = []
        self.selected = -1
        self.shape = "rectangle"
        self.editing = False
        self._draft: list[list[float]] = []
        self._dragging = False
        self._vertex: int | None = None
        self.setMouseTracking(True)

    def set_image(self, image: np.ndarray | None) -> None:
        self.image = image
        self._draft = []
        if image is None:
            self.pixmap = None
        else:
            rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            self.pixmap = QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy())
        self.update()

    def image_rect(self) -> QRectF:
        if self.pixmap is None:
            return QRectF()
        scale = min(self.width() / self.pixmap.width(), self.height() / self.pixmap.height())
        width, height = self.pixmap.width() * scale, self.pixmap.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def _normalized(self, point: QPointF, clamp: bool = False) -> list[float] | None:
        rect = self.image_rect()
        if rect.isEmpty() or (not clamp and not rect.contains(point)):
            return None
        return [max(0.0, min(1.0, (point.x() - rect.left()) / rect.width())),
                max(0.0, min(1.0, (point.y() - rect.top()) / rect.height()))]

    def _screen(self, point: list[float]) -> QPointF:
        rect = self.image_rect()
        return QPointF(rect.left() + point[0] * rect.width(), rect.top() + point[1] * rect.height())

    def _draw_region(self, painter: QPainter, definition: dict, color: QColor, selected: bool = False) -> None:
        points = [self._screen(p) for p in definition.get("points", [])]
        if not points:
            return
        painter.setPen(QPen(color, 2.5 if selected else 1.5))
        fill = QColor(color)
        fill.setAlpha(22)
        painter.setBrush(fill)
        shape = definition.get("shape", "rectangle")
        if shape == "rectangle" and len(points) >= 2:
            painter.drawRect(QRectF(points[0], points[1]).normalized())
        elif shape == "circle" and len(points) >= 2:
            radius = math.hypot(points[1].x() - points[0].x(), points[1].y() - points[0].y())
            painter.drawEllipse(points[0], radius, radius)
        elif shape == "polygon" and len(points) >= 2:
            painter.drawPolygon(QPolygonF(points))
        if selected:
            painter.setBrush(QColor("#ffffff"))
            for point in points:
                painter.drawEllipse(point, 4, 4)
        painter.setPen(color)
        painter.drawText(points[0] + QPointF(7, -7), str(definition.get("name", "")))

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#050806"))
        if self.pixmap is None:
            painter.setPen(QColor("#a7bdad"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load a representative native image or camera frame to draw ROIs")
            return
        painter.drawPixmap(self.image_rect().toRect(), self.pixmap)
        for index, definition in enumerate(self.regions):
            color = QColor("#ffc857" if definition.get("exclusion") else "#00E676")
            self._draw_region(painter, definition, color, index == self.selected)
        if self._draft:
            self._draw_region(painter, {"shape": self.shape, "points": self._draft}, QColor("#45ccff"), True)
        painter.end()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self._draft = []
            self.update()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self._normalized(event.position())
        if point is None:
            return
        if self.editing:
            if 0 <= self.selected < len(self.regions):
                for index, vertex in enumerate(self.regions[self.selected].get("points", [])):
                    position = self._screen(vertex)
                    if math.hypot(event.position().x() - position.x(), event.position().y() - position.y()) <= 14:
                        self._vertex = index
                        self._dragging = True
                        break
            return
        if self.shape == "polygon":
            self._draft.append(point)
        else:
            self._draft = [point, point.copy()]
            self._dragging = True
        self.update()

    def mouseMoveEvent(self, event: Any) -> None:
        if not self._dragging:
            return
        point = self._normalized(event.position(), clamp=True)
        if point is None:
            return
        if self.editing and self._vertex is not None and 0 <= self.selected < len(self.regions):
            self.regions[self.selected]["points"][self._vertex] = point
            self.regions_changed.emit()
        elif self._draft:
            self._draft[-1] = point
        self.update()

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._dragging:
            return
        self._dragging = False
        if self.editing:
            self._vertex = None
        elif self.shape != "polygon":
            self.finish_drawing()

    def mouseDoubleClickEvent(self, event: Any) -> None:
        if self.shape == "polygon" and not self.editing:
            self.finish_drawing()

    def finish_drawing(self) -> None:
        points = list(self._draft)
        # A double click can repeat the final vertex; remove adjacent duplicates.
        points = [p for i, p in enumerate(points) if i == 0 or p != points[i - 1]]
        needed = 3 if self.shape == "polygon" else 2
        if len(points) < needed:
            return
        if self.shape != "polygon":
            p0, p1 = self._screen(points[0]), self._screen(points[1])
            if math.hypot(p1.x() - p0.x(), p1.y() - p0.y()) < 6:
                self._draft = []
                self.update()
                return
        self.region_created.emit({"shape": self.shape, "points": points})
        self._draft = []
        self.update()


class ROIEditor(QDialog):
    def __init__(self, service: Any, view: str = "inner_face", image: np.ndarray | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.service = service
        self._dirty = False
        self._current_view = view
        self._updating = False
        self.setWindowTitle("ROI Editor · product and view masks")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(1240, 850)
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.view = QComboBox()
        for key, title in VIEW_LABELS.items():
            self.view.addItem(title, key)
        self.view.setCurrentIndex(max(0, self.view.findData(view)))
        toolbar.addWidget(QLabel("View"))
        toolbar.addWidget(self.view)
        toolbar.addStretch()
        load = QPushButton("Load Image…")
        load.clicked.connect(self.load_image)
        live = QPushButton("Use Latest Camera Frame")
        live.clicked.connect(self.use_camera)
        toolbar.addWidget(load)
        toolbar.addWidget(live)
        layout.addLayout(toolbar)
        self.instructions = QLabel("Rectangle / circle: drag across the image. Polygon: click vertices, double click or Finish Polygon to close. Right click cancels drawing. In edit mode, drag the selected region's white vertex handles.")
        self.instructions.setWordWrap(True)
        layout.addWidget(self.instructions)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.canvas = ROICanvas()
        self.canvas.set_image(image)
        self.canvas.region_created.connect(self.add_region)
        self.canvas.regions_changed.connect(self.changed)
        split.addWidget(self.canvas)
        panel = QWidget()
        panel.setMaximumWidth(330)
        controls = QVBoxLayout(panel)
        form = QFormLayout()
        self.shape = QComboBox()
        for title, key in [("Rectangle", "rectangle"), ("Circle", "circle"), ("Polygon", "polygon")]:
            self.shape.addItem(title, key)
        self.shape.currentIndexChanged.connect(self.change_tool)
        form.addRow("Draw shape", self.shape)
        self.name = QLineEdit("Inspection region")
        self.name.textEdited.connect(self.update_selected)
        form.addRow("Region name", self.name)
        self.role = QComboBox()
        for title, key in [("Inspection area", "inspection"), ("Rib annulus · outer", "rib_annulus_outer"), ("Rib annulus · inner", "rib_annulus_inner"), ("Mounting structure", "mounting"), ("Rim area", "rim"), ("Inner machined face", "inner_face")]:
            self.role.addItem(title, key)
        self.role.currentIndexChanged.connect(self.update_selected)
        form.addRow("Region purpose", self.role)
        self.exclude = QCheckBox("Exclude / keep-out zone")
        self.exclude.toggled.connect(self.update_selected)
        form.addRow(self.exclude)
        self.edit = QCheckBox("Edit selected vertices")
        self.edit.toggled.connect(self.change_tool)
        form.addRow(self.edit)
        controls.addLayout(form)
        finish = QPushButton("Finish Polygon")
        finish.clicked.connect(self.canvas.finish_drawing)
        controls.addWidget(finish)
        self.regions = QListWidget()
        self.regions.currentRowChanged.connect(self.select_region)
        controls.addWidget(self.regions, 1)
        delete = QPushButton("Delete Selected Region")
        delete.clicked.connect(self.delete_selected)
        controls.addWidget(delete)
        clear = QPushButton("Clear All Regions")
        clear.clicked.connect(self.clear_regions)
        controls.addWidget(clear)
        hint = QLabel("Green regions are included; amber regions are excluded. For a rib annulus, draw an outer inspection circle and a smaller central exclusion circle. Draw mounting keep-outs separately. Masks and coordinates are stored per product and view.")
        hint.setWordWrap(True)
        controls.addWidget(hint)
        split.addWidget(panel)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)
        self.status = QLabel("Normalized coordinates · source pixels are preserved")
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.view.currentIndexChanged.connect(self.change_view)
        self.load_regions()

    def change_tool(self, *_: Any) -> None:
        self.canvas.shape = str(self.shape.currentData())
        self.canvas.editing = self.edit.isChecked()
        self.canvas._draft = []
        self.canvas.update()

    def load_regions(self) -> None:
        try:
            self.canvas.regions = deepcopy(self.service.get_rois(self._current_view))
            self.canvas.selected = -1
            self.refresh_list()
            self._dirty = False
            self.status.setText("Loaded saved masks for " + VIEW_LABELS.get(self._current_view, self._current_view))
        except Exception as exc:
            QMessageBox.warning(self, "ROI configuration", str(exc))

    def refresh_list(self) -> None:
        selected = self.canvas.selected
        self._updating = True
        self.regions.clear()
        for definition in self.canvas.regions:
            prefix = "EXCLUDE" if definition.get("exclusion") else "INCLUDE"
            self.regions.addItem(f"{prefix} · {definition.get('name', definition.get('shape', 'Region'))}")
        if 0 <= selected < self.regions.count():
            self.regions.setCurrentRow(selected)
        self._updating = False
        self.canvas.update()

    def changed(self) -> None:
        self._dirty = True
        self.status.setText("Unsaved region edits · click Save to apply")

    def add_region(self, definition: dict) -> None:
        definition.update(name=self.name.text().strip() or f"Region {len(self.canvas.regions) + 1}",
                          exclusion=self.exclude.isChecked(), role=self.role.currentData())
        self.canvas.regions.append(definition)
        self.canvas.selected = len(self.canvas.regions) - 1
        self.changed()
        self.refresh_list()

    def select_region(self, index: int) -> None:
        if self._updating:
            return
        self.canvas.selected = index
        if 0 <= index < len(self.canvas.regions):
            definition = self.canvas.regions[index]
            self._updating = True
            self.name.setText(str(definition.get("name", "Region")))
            self.role.setCurrentIndex(max(0, self.role.findData(definition.get("role", "inspection"))))
            self.exclude.setChecked(bool(definition.get("exclusion")))
            self._updating = False
        self.canvas.update()

    def update_selected(self, *_: Any) -> None:
        if self._updating:
            return
        index = self.canvas.selected
        if 0 <= index < len(self.canvas.regions):
            self.canvas.regions[index].update(name=self.name.text().strip() or "Region",
                exclusion=self.exclude.isChecked(), role=self.role.currentData())
            self.changed()
            self.refresh_list()

    def delete_selected(self) -> None:
        index = self.canvas.selected
        if 0 <= index < len(self.canvas.regions):
            del self.canvas.regions[index]
            self.canvas.selected = -1
            self.changed()
            self.refresh_list()

    def clear_regions(self) -> None:
        self.canvas.regions = []
        self.canvas.selected = -1
        self.changed()
        self.refresh_list()

    def change_view(self) -> None:
        if self._dirty:
            # Preserve edits automatically in their original view before switching.
            if not self.save(show_message=False):
                self.view.blockSignals(True)
                self.view.setCurrentIndex(max(0, self.view.findData(self._current_view)))
                self.view.blockSignals(False)
                return
        self._current_view = str(self.view.currentData())
        self.load_regions()

    def load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load a representative image for the selected face", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            try:
                image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("The file is not a readable image")
                self.canvas.set_image(image)
            except Exception as exc:
                QMessageBox.warning(self, "ROI image", str(exc))

    def use_camera(self) -> None:
        try:
            packet = self.service.preview()
            if packet is None or not isinstance(packet.get("image"), np.ndarray):
                raise ValueError("No current camera frame. Start acquisition before opening the ROI editor.")
            self.canvas.set_image(packet["image"])
        except Exception as exc:
            QMessageBox.warning(self, "Camera frame", str(exc))

    def save(self, *_: Any, show_message: bool = True) -> bool:
        try:
            self.service.save_rois(self._current_view, deepcopy(self.canvas.regions))
            self._dirty = False
            self.status.setText("Masks saved for " + VIEW_LABELS.get(self._current_view, self._current_view) + ". Revalidate trained models after changing ROIs.")
            return True
        except Exception as exc:
            QMessageBox.warning(self, "ROI save failed", str(exc))
            return False

    def reject(self) -> None:
        if self._dirty:
            choice = QMessageBox.question(self, "Unsaved masks", "Save the current ROI edits before closing?", QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
            if choice == QMessageBox.StandardButton.Cancel:
                return
            if choice == QMessageBox.StandardButton.Save and not self.save():
                return
        super().reject()

    def closeEvent(self, event: Any) -> None:
        if self._dirty:
            event.ignore()
            self.reject()
        else:
            event.accept()
