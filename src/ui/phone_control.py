"""Opt-in pairing and lifecycle controls for a phone on the local Wi-Fi."""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QComboBox, QDialog, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout)


def pairing_qr_image(link: str) -> np.ndarray | None:
    """Encode locally; the pairing secret is never sent to an external service."""
    try:
        pixels = cv2.QRCodeEncoder_create().encode(link)
    except (AttributeError, cv2.error):
        return None
    pixels = np.pad(pixels, 4, constant_values=255)
    scale = max(1, 260 // pixels.shape[0])
    return cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


class PhoneControlDialog(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Phone Control")
        self.setMinimumWidth(540)
        layout = QVBoxLayout(self)
        instructions = QLabel("Connect the phone and this PC to the same Wi-Fi. Start phone control, "
                              "scan the QR code or open the pairing link in the phone browser. "
                              "Close this setup window before sending decisions.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        row = QHBoxLayout()
        row.addWidget(QLabel("PC Wi-Fi address"))
        self.address = QComboBox()
        self.address.setEditable(True)
        from src.remote.server import local_ipv4_addresses
        self.address.addItems([address for address in local_ipv4_addresses() if not address.startswith("127.")])
        self.address.setMinimumWidth(180)
        self.address.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.address.lineEdit().setPlaceholderText("Enter the PC's Wi-Fi IPv4 address")
        row.addWidget(self.address, 1)
        row.addWidget(QLabel("Port"))
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.setValue(window.phone_server.port if window.phone_server is not None and window.phone_server.running else 8765)
        row.addWidget(self.port)
        layout.addLayout(row)
        self.link = QLineEdit()
        self.link.setReadOnly(True)
        self.link.setPlaceholderText("Start phone control to create a pairing link")
        layout.addWidget(self.link)
        self.qrcode_image: np.ndarray | None = None
        self.qrcode = QLabel()
        self.qrcode.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.qrcode.setStyleSheet("background:white;padding:8px;")
        layout.addWidget(self.qrcode, 0, Qt.AlignmentFlag.AlignHCenter)
        buttons = QHBoxLayout()
        self.start = QPushButton("Start Phone Control")
        self.start.clicked.connect(self.start_server)
        buttons.addWidget(self.start)
        self.stop = QPushButton("Stop Phone Control")
        self.stop.clicked.connect(self.stop_server)
        buttons.addWidget(self.stop)
        self.copy = QPushButton("Copy Link")
        self.copy.clicked.connect(self.copy_link)
        buttons.addWidget(self.copy)
        layout.addLayout(buttons)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        note = QLabel("On Windows, allow the application through the firewall for a Private network. "
                      "Use the PC's Wi-Fi address rather than 127.0.0.1. "
                      "The pairing link grants control until you stop or restart phone control; "
                      "share it only with the inspector.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        layout.addWidget(note)
        close = QPushButton("Close Setup")
        close.clicked.connect(self.accept)
        layout.addWidget(close)
        self.address.currentTextChanged.connect(self.update_link)
        self.refresh()

    def refresh(self) -> None:
        server = self.window.phone_server
        running = server is not None and server.running
        self.start.setEnabled(not running)
        self.stop.setEnabled(running)
        self.port.setEnabled(not running)
        self.status.setText(f"Phone control is running on port {server.port}. Close this setup window to enable decisions." if running else
                            "Phone control is stopped.")
        self.update_link()

    def update_link(self) -> None:
        server = self.window.phone_server
        link = ""
        if server is not None and server.running:
            try:
                link = server.pairing_url(self.address.currentText().strip())
            except ValueError:
                pass
        self.link.setText(link)
        self.copy.setEnabled(bool(link))
        self.qrcode_image = pairing_qr_image(link) if link else None
        if self.qrcode_image is None:
            self.qrcode.clear()
            self.qrcode.hide()
        else:
            pixels = self.qrcode_image
            image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0],
                           QImage.Format.Format_Grayscale8).copy()
            self.qrcode.setPixmap(QPixmap.fromImage(image))
            self.qrcode.show()

    def start_server(self) -> None:
        try:
            self.window.start_phone_control(self.port.value())
        except (OSError, ValueError, RuntimeError) as exc:
            self.status.setText(f"Could not start phone control: {exc}")
            return
        self.refresh()

    def stop_server(self) -> None:
        self.window.stop_phone_control()
        self.refresh()

    def copy_link(self) -> None:
        if self.link.text():
            QApplication.clipboard().setText(self.link.text())
            self.status.setText("Pairing link copied. Open it on the phone, then close this setup window.")
