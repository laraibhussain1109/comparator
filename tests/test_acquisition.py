"""Software contracts with mock capture devices, not camera acceptance tests."""
from contextlib import nullcontext
from types import SimpleNamespace
import sys
import time

import cv2
import numpy as np
import pytest

from src.acquisition import (CameraConfig, CameraService, FramePacket,
                             QualityChecker, runtime_diagnostics)


class Capture:
    def __init__(self, image=None, *, opened=True, fail_after=None):
        self.image = np.full((64, 96, 3), 120, np.uint8) if image is None else image
        self.opened = opened
        self.fail_after = fail_after
        self.reads = 0
        self.settings = []
        self.values = {cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
                       cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_WIDTH: 96,
                       cv2.CAP_PROP_FRAME_HEIGHT: 64}

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.settings.append((prop, value))
        if prop == cv2.CAP_PROP_EXPOSURE:
            return False
        # This mock deliberately ignores unsupported requested resolution.
        if prop not in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            self.values[prop] = value
        return True

    def get(self, prop):
        return self.values.get(prop, 0)

    def read(self):
        time.sleep(.005)
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        return True, self.image

    def release(self):
        self.opened = False


def until(condition, seconds=2):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(.005)
    pytest.fail("Camera state did not arrive in time")


def test_camera_negotiates_mjpeg_before_resolution_and_keeps_native_copy():
    capture = Capture()
    service = CameraService(CameraConfig(width=96, height=64), lambda *_: capture)
    service.start()
    try:
        until(lambda: service.diagnostics()["connected"])
        first = service.latest()
        assert first.image.shape == (64, 96, 3)
        assert first.timestamp > 0 and first.sequence >= 1
        first.image[:] = 0
        assert service.latest().image[0, 0, 0] == 120
        assert capture.settings[:3] == [
            (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")),
            (cv2.CAP_PROP_FRAME_WIDTH, 96), (cv2.CAP_PROP_FRAME_HEIGHT, 64)]
        assert service.diagnostics()["format_verified"]
        until(lambda: service.diagnostics()["captured_frames"] >= 4)
        diag = service.diagnostics()
        assert diag["actual_fps"] > 0
        assert diag["dropped_frames"] >= 1
        assert service.latest().sequence > first.sequence
    finally:
        service.stop()
    assert service.latest() is None and service.diagnostics()["status"] == "stopped"
    assert not capture.opened


def test_camera_actual_dimensions_override_requested_driver_settings():
    service = CameraService(CameraConfig(), lambda *_: Capture())
    service.start()
    try:
        until(lambda: service.diagnostics()["connected"])
        diag = service.diagnostics()
        assert diag["requested_resolution"] == [3840, 2160]
        assert diag["actual_resolution"] == [96, 64]
        assert diag["status"] == "streaming_degraded"
        assert not diag["format_verified"] and diag["warnings"]
    finally:
        service.stop()


def test_camera_backend_fallback_never_changes_selected_device(monkeypatch):
    monkeypatch.setattr("src.acquisition.camera.platform.system", lambda: "Windows")
    calls = []

    def factory(index, backend):
        calls.append((index, backend))
        return Capture(opened=backend != cv2.CAP_DSHOW)

    service = CameraService(CameraConfig(index=3, width=96, height=64), factory)
    service.start()
    try:
        until(lambda: service.diagnostics()["connected"])
        assert calls == [(3, cv2.CAP_DSHOW), (3, cv2.CAP_MSMF)]
        assert service.diagnostics()["backend"] == "msmf"
    finally:
        service.stop()


def test_camera_disconnect_reconnect_and_controls_verified_by_readback():
    calls = []

    def factory(index, backend):
        calls.append(index)
        return Capture(fail_after=2 if len(calls) == 1 else None)

    service = CameraService(CameraConfig(index=2, width=96, height=64, reconnect_interval=.01), factory)
    service.start()
    try:
        until(lambda: service.diagnostics()["reconnects"] >= 1 and service.diagnostics()["connected"])
        assert calls == [2, 2]
        assert service.diagnostics()["read_failures"] == 1
        assert service.set_control("gain", 2)
        assert not service.set_control("exposure", -6)
        assert not service.set_control("nonexistent_control", 1)
        assert not service.set_control("gain", float("nan"))
        controls = service.diagnostics()["controls"]
        assert controls["gain"]["supported"] is True
        assert controls["exposure"]["supported"] is False
        assert controls["exposure"]["reason"]
    finally:
        service.stop()


def test_unavailable_camera_has_explicit_status_and_no_frames():
    calls = []

    def factory(index, backend):
        calls.append(index)
        return Capture(opened=False)

    service = CameraService(CameraConfig(index=4, backend="any", reconnect_interval=.1), factory)
    service.start()
    try:
        until(lambda: service.diagnostics()["status"] == "disconnected")
        assert service.latest() is None
        assert "device did not open" in service.diagnostics()["error"]
        assert set(calls) == {4}
    finally:
        service.stop()


def texture():
    checker = ((np.indices((96, 128)).sum(axis=0) // 4) % 2 * 100 + 70).astype(np.uint8)
    return cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)


def test_quality_rejects_invalid_blurred_and_exposure_clipped_images():
    checker = QualityChecker()
    assert not checker.check(None).valid
    assert checker.check(texture()).valid
    blurred = cv2.GaussianBlur(texture(), (21, 21), 5)
    assert "insufficient_focus" in checker.check(blurred).reasons
    assert "underexposed" in checker.check(np.zeros((32, 32, 3), np.uint8)).reasons
    clipped = checker.check(np.full((32, 32, 3), 255, np.uint8))
    assert not clipped.valid and "excessive_highlight_clipping" in clipped.reasons


def test_quality_mask_part_presence_completeness_and_pose():
    checker = QualityChecker({"require_mask": True})
    image = texture()
    assert "part_mask_required" in checker.check(image).reasons
    absent = checker.check(image, np.zeros(image.shape[:2], np.uint8))
    assert not absent.valid and "part_absent_or_too_small" in absent.reasons
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(mask, (64, 48), 30, 255, -1)
    assert checker.check(image, mask).valid
    assert "uncertain_pose" in checker.check(image, mask, pose_confidence=.2).reasons
    mask[:, :10] = 255
    assert "part_clipped_at_frame_edge" in checker.check(image, mask).reasons
    assert not checker.check(image, mask[:3]).valid


def test_quality_motion_and_stale_previous_shapes():
    image = texture()
    checker = QualityChecker({"max_motion": .01})
    assert checker.check(image, previous=image).metrics["motion"] == 0
    moved = np.roll(image, 3, axis=1)
    assert "part_moving" in checker.check(moved, previous=image).reasons
    assert "previous_frame_shape_mismatch" in checker.check(image, previous=image[:20]).reasons


def test_select_sharpest_requires_recent_unique_stable_valid_frames():
    checker = QualityChecker({"max_motion": .8, "stable_frames": 3})
    image = texture()
    soft = cv2.GaussianBlur(image, (3, 3), .5)
    packets = [FramePacket(soft, 1, 9.5), FramePacket(image, 2, 9.6), FramePacket(soft, 3, 9.7)]
    selected = checker.select_sharpest(packets, now=10)
    assert selected.sequence == 2
    selected.image[:] = 0
    assert np.any(packets[1].image)
    assert checker.select_sharpest(packets, now=20) is None
    assert checker.select_sharpest([packets[0]] * 3, now=10) is None
    assert checker.select_sharpest(packets + [FramePacket(np.zeros_like(image), 4, 9.8)], now=10) is None


def test_selection_rechecks_freshness_after_native_quality_processing(monkeypatch):
    import src.acquisition.quality as quality_module
    ticks = iter([10., 12.])
    monkeypatch.setattr(quality_module, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    packets = [FramePacket(texture(), index, 9.5 + index * .1) for index in range(3)]
    assert QualityChecker().select_sharpest(packets) is None


def test_discontinuous_capture_history_cannot_establish_stability():
    image = texture()
    packets = [FramePacket(image, 1, 9.1), FramePacket(image, 2, 9.7), FramePacket(image, 3, 9.8)]
    assert QualityChecker().select_sharpest(packets, now=10) is None


def test_equal_sharpness_prefers_freshest_stable_capture():
    image = texture()
    packets = [FramePacket(image, index, 9.5 + index * .1) for index in range(3)]
    assert QualityChecker().select_sharpest(packets, now=10).sequence == 2


def test_diagnostics_torch_optional_and_cpu_truthful(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    missing = runtime_diagnostics()
    assert not missing["torch_available"] and missing["device"] == "cpu"
    assert missing["inference_device"] == "not_loaded"
    torch = SimpleNamespace(version=SimpleNamespace(cuda=None), cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch)
    cpu = runtime_diagnostics()
    assert cpu["torch_available"] and not cpu["cuda_available"]
    assert not cpu["cuda_compute_verified"] and cpu["gpu_name"] is None


def test_cuda_driver_availability_does_not_hide_kernel_incompatibility(monkeypatch):
    def no_kernel(*args, **kwargs):
        raise RuntimeError("no kernel image is available for execution on the device")

    cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
                           get_device_properties=lambda _: SimpleNamespace(name="Test GPU", major=12, minor=0),
                           get_arch_list=lambda: ["sm_80"])
    torch = SimpleNamespace(version=SimpleNamespace(cuda="12.4"), cuda=cuda,
                            inference_mode=nullcontext, ones=no_kernel)
    monkeypatch.setitem(sys.modules, "torch", torch)
    result = runtime_diagnostics()
    assert result["cuda_available"] and not result["cuda_compute_verified"]
    assert result["device"] == "cpu" and "no kernel image" in result["error"]


def test_invalid_camera_configuration_is_rejected():
    with pytest.raises(ValueError):
        CameraConfig(index=-1)
    with pytest.raises(ValueError):
        CameraConfig(backend="unavailable")
