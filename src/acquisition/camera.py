"""Latest-frame-only OpenCV camera provider.

Requested settings are preferences, never evidence of the camera's actual
format. Frame dimensions and driver-reported FOURCC are reported separately.
Backends may be retried, but the selected camera index never changes.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import math
import platform
import threading
import time
from typing import Callable, Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 3840
    height: int = 2160
    fps: float = 30
    backend: str = "auto"
    reconnect_interval: float = 1.0

    def __post_init__(self) -> None:
        if self.index < 0 or self.width <= 0 or self.height <= 0 or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Camera index, dimensions and FPS must be valid positive settings")
        if self.backend.lower() not in {"auto", "dshow", "msmf", "any", "v4l2"}:
            raise ValueError(f"Unsupported capture backend: {self.backend}")
        if not math.isfinite(self.reconnect_interval) or self.reconnect_interval <= 0:
            raise ValueError("Reconnect interval must be positive")


@dataclass(frozen=True)
class FramePacket:
    image: np.ndarray
    sequence: int
    timestamp: float


_CONTROLS = {
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "gain": cv2.CAP_PROP_GAIN,
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION,
    "sharpness": cv2.CAP_PROP_SHARPNESS,
    "focus": cv2.CAP_PROP_FOCUS,
    "autofocus": cv2.CAP_PROP_AUTOFOCUS,
    "white_balance": cv2.CAP_PROP_WB_TEMPERATURE,
    "auto_white_balance": cv2.CAP_PROP_AUTO_WB,
    "gamma": cv2.CAP_PROP_GAMMA,
    "zoom": cv2.CAP_PROP_ZOOM,
}


def _fourcc(value: float) -> str:
    if not math.isfinite(value) or value <= 0:
        return "unknown"
    code = int(value)
    decoded = "".join(chr((code >> (8 * i)) & 255) for i in range(4))
    return decoded if all(32 <= ord(c) <= 126 for c in decoded) else "unknown"


class CameraService:
    def __init__(self, config: CameraConfig | None = None,
                 capture_factory: Callable[[int, int], Any] | None = None) -> None:
        self.config = config or CameraConfig()
        self._factory = capture_factory or cv2.VideoCapture
        self._lock = threading.RLock()
        self._io_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: Any | None = None
        self._latest: FramePacket | None = None
        self._sequence = 0
        self._consumed_sequence = 0
        self._timestamps: deque[float] = deque(maxlen=120)
        self._diag: dict[str, Any] = {
            "status": "stopped", "connected": False, "device_index": self.config.index,
            "device_identity": None, "identity_note": "OpenCV identifies this device by index; reselect after USB topology changes.",
            "requested_resolution": [self.config.width, self.config.height],
            "requested_fps": self.config.fps, "requested_fourcc": "MJPG",
            "actual_resolution": None, "driver_resolution": None,
            "driver_fps": None, "fourcc": "unknown", "backend": None,
            "resolution_matches_request": False, "format_verified": False,
            "captured_frames": 0, "dropped_frames": 0, "read_failures": 0,
            "reconnects": 0, "error": None, "warnings": [], "controls": {},
        }

    def start(self) -> None:
        """Begin acquisition without blocking the GUI on camera open/read."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._diag.update(status="connecting", error=None)
            self._thread = threading.Thread(target=self._run, name=f"camera-{self.config.index}", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))
        with self._lock:
            self._latest = None
            self._diag["connected"] = False
            if thread and thread.is_alive():
                # Calling release concurrently with a blocked native read can
                # crash some drivers. Keep the daemon isolated until it exits.
                self._diag.update(status="stop_timeout", error="Camera driver is still blocked in read; acquisition stop requested.")
            else:
                self._diag["status"] = "stopped"

    def latest(self) -> FramePacket | None:
        """Return an owned native-size BGR copy; disconnected frames are never reused."""
        with self._lock:
            if self._stop.is_set() or not self._diag["connected"] or self._latest is None:
                return None
            frame = self._latest
            self._consumed_sequence = frame.sequence
            return FramePacket(frame.image.copy(), frame.sequence, frame.timestamp)

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._diag)
            result["warnings"] = list(self._diag["warnings"])
            result["controls"] = {name: dict(info) for name, info in self._diag["controls"].items()}
            times = list(self._timestamps)
            now = time.monotonic()
            result["actual_fps"] = ((len(times) - 1) / (times[-1] - times[0])
                                    if len(times) > 1 and times[-1] > times[0] and now - times[-1] < 2 else 0.0)
            result["last_frame_age_seconds"] = now - times[-1] if times else None
            result["capture_thread_alive"] = bool(self._thread and self._thread.is_alive())
            return result

    def set_control(self, name: str, value: float) -> bool:
        """Apply a raw backend value and verify readback, rather than pretend support.

        Auto-exposure values differ between DirectShow and Media Foundation;
        callers must use values supported by the selected backend. A zero getter
        alone is not treated as evidence that a control is supported.
        """
        name = name.lower()
        if name not in _CONTROLS or not math.isfinite(value):
            return False
        with self._io_lock:
            capture = self._capture
            if capture is None or not capture.isOpened():
                return False
            try:
                accepted = bool(capture.set(_CONTROLS[name], float(value)))
                readback = float(capture.get(_CONTROLS[name]))
                verified = accepted and math.isfinite(readback) and math.isclose(readback, value, abs_tol=.05, rel_tol=.02)
                info = {"supported": verified, "requested": value,
                        "readback": readback if math.isfinite(readback) else None,
                        "reason": None if verified else "Driver rejected control or did not confirm requested value."}
            except Exception as exc:
                verified = False
                info = {"supported": False, "requested": value, "readback": None, "reason": str(exc)}
            with self._lock:
                self._diag["controls"][name] = info
            return verified

    def _backends(self) -> list[tuple[str, int]]:
        options = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY, "v4l2": cv2.CAP_V4L2}
        if self.config.backend.lower() != "auto":
            name = self.config.backend.lower()
            return [(name, options[name])]
        names = ["dshow", "msmf", "any"] if platform.system() == "Windows" else ["any"]
        return [(name, options[name]) for name in names]

    def _open(self) -> tuple[Any, np.ndarray, str] | None:
        errors = []
        for name, backend in self._backends():
            if self._stop.is_set():
                break
            capture = None
            try:
                capture = self._factory(self.config.index, backend)
                if not capture.isOpened():
                    raise RuntimeError("device did not open")
                # UVC devices commonly require MJPEG before requesting 4K.
                for prop, value in (
                    (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")),
                    (cv2.CAP_PROP_FRAME_WIDTH, self.config.width),
                    (cv2.CAP_PROP_FRAME_HEIGHT, self.config.height),
                    (cv2.CAP_PROP_FPS, self.config.fps),
                    (cv2.CAP_PROP_BUFFERSIZE, 1),
                ):
                    capture.set(prop, value)
                ok, image = capture.read()
                if not ok or not self._valid_frame(image):
                    raise RuntimeError("device produced no valid BGR frame")
                return capture, image, name
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                if capture is not None:
                    self._release(capture)
        with self._lock:
            self._diag.update(status="disconnected", connected=False, error="; ".join(errors) or "Acquisition stopped")
        return None

    @staticmethod
    def _valid_frame(image: Any) -> bool:
        return isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3 and image.size > 0

    @staticmethod
    def _release(capture: Any) -> None:
        try:
            capture.release()
        except Exception as exc:
            LOGGER.warning("Camera driver failed to release: %s", exc)

    def _publish(self, image: np.ndarray) -> None:
        timestamp = time.monotonic()
        height, width = image.shape[:2]
        with self._lock:
            if self._stop.is_set():
                return
            if self._latest is not None and self._latest.sequence > self._consumed_sequence:
                self._diag["dropped_frames"] += 1
            self._sequence += 1
            self._latest = FramePacket(image.copy(), self._sequence, timestamp)
            self._timestamps.append(timestamp)
            self._diag["captured_frames"] += 1
            self._diag["actual_resolution"] = [width, height]
            matched = width == self.config.width and height == self.config.height
            self._diag["resolution_matches_request"] = matched
            self._diag["format_verified"] = matched and self._diag["fourcc"] == "MJPG"
            warnings = []
            if not matched:
                warnings.append(f"Requested {self.config.width}×{self.config.height}; received {width}×{height}.")
            if self._diag["fourcc"] != "MJPG":
                warnings.append(f"MJPEG not confirmed by driver (reported {self._diag['fourcc']}).")
            self._diag["warnings"] = warnings
            self._diag.update(status="streaming" if not warnings else "streaming_degraded", connected=True, error=None)

    def _run(self) -> None:
        connected_once = False
        try:
            while not self._stop.is_set():
                opened = self._open()
                if opened is None:
                    self._stop.wait(self.config.reconnect_interval)
                    continue
                capture, frame, backend = opened
                try:
                    with self._io_lock:
                        self._capture = capture
                        with self._lock:
                            self._timestamps.clear()
                            self._diag["backend"] = backend
                            self._diag["fourcc"] = _fourcc(float(capture.get(cv2.CAP_PROP_FOURCC)))
                            self._diag["driver_fps"] = float(capture.get(cv2.CAP_PROP_FPS))
                            self._diag["driver_resolution"] = [float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
                            self._diag["controls"] = {name: {"supported": None, "readback": None, "reason": "Support unverified until set and read back."} for name in _CONTROLS}
                            self._diag["reconnects"] += int(connected_once)
                    connected_once = True
                    while not self._stop.is_set():
                        self._publish(frame)
                        with self._io_lock:
                            ok, frame = capture.read()
                        if not ok or not self._valid_frame(frame):
                            with self._lock:
                                self._diag["read_failures"] += 1
                            raise RuntimeError("Camera disconnected or returned an invalid frame")
                except Exception as exc:
                    LOGGER.warning("Camera %s: %s", self.config.index, exc)
                    with self._lock:
                        self._latest = None
                        self._diag.update(status="disconnected", connected=False, error=str(exc))
                finally:
                    with self._io_lock:
                        self._capture = None
                        self._release(capture)
                if not self._stop.is_set():
                    self._stop.wait(self.config.reconnect_interval)
        finally:
            with self._lock:
                self._latest = None
                self._diag.update(status="stopped", connected=False)
