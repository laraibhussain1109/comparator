"""Live-capture gates with software housings and an explicitly mocked camera.

These fixtures validate provenance, freshness and session coverage contracts;
they are not camera acceptance tests or industrial defect-accuracy evidence.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.acquisition import FramePacket
from src.inspection.session import ViewDecision
from src.service import NeuroIrisService


class Clock:
    def __init__(self, now=100.):
        self.now = now

    def monotonic(self):
        return self.now


class MockCamera:
    """Return independent copies like the real CameraService.latest()."""
    def __init__(self):
        self.packet = None
        self.connected = True

    def latest(self):
        packet = self.packet
        return (FramePacket(packet.image.copy(), packet.sequence, packet.timestamp)
                if packet is not None else None)

    def diagnostics(self):
        return {"connected": self.connected,
                "status": "streaming" if self.connected else "disconnected"}

    def stop(self):
        self.connected = False


def housing():
    """A textured circular fixture distinguishable from the blue background."""
    image = np.full((256, 320, 3), (155, 95, 25), np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(mask, (160, 128), 87, 255, -1)
    noise = np.random.default_rng(7).integers(100, 195, image.shape[:2], dtype=np.uint8)
    image[mask > 0] = np.repeat(noise[:, :, None], 3, axis=2)[mask > 0]
    return image


@pytest.fixture
def live_service(tmp_path, monkeypatch):
    import src.service as service_module
    import src.acquisition.quality as quality_module

    clock = Clock()
    monkeypatch.setattr(service_module, "time", clock)
    monkeypatch.setattr(quality_module, "time", clock)
    service = NeuroIrisService({
        "product": "test_housing",
        "paths": {name: tmp_path / name for name in ("dataset", "artifacts", "results", "runtime")},
        "views": {"required": ["inner_face", "ribbed_face"], "enabled": ["inner_face", "ribbed_face"]},
        "models": {"backend": "geometry"}, "quality": {},
    })
    service.camera = MockCamera()
    inferred = []

    def infer(image, view, backend=None):
        inferred.append((image.copy(), view))
        return (ViewDecision(view, "PASS", ["Mock inference passed"],
                             metrics={"inspection_scope": "geometry_only"}), image.copy(), None)

    monkeypatch.setattr(service, "_infer", infer)
    service.begin_part("physical-001")
    return service, service.camera, clock, inferred


def preview_frames(service, camera, clock, images, *, first_sequence=1,
                   timestamps=None):
    timestamps = timestamps or [100.1 + .1 * index for index in range(len(images))]
    previews = []
    for sequence, image, timestamp in zip(range(first_sequence, first_sequence + len(images)), images, timestamps):
        camera.packet = FramePacket(image, sequence, timestamp)
        clock.now = timestamp
        previews.append(service.preview())
    clock.now += .05
    return previews


def assert_no_coverage(service, inferred):
    assert inferred == []
    assert service.session.decisions == {}
    assert service.session.missing_views == ["inner_face", "ribbed_face"]
    assert service.session.preview().verdict == "RECHECK"
    assert not service.session.counted
    assert service.frames == {}


def test_live_inspection_uses_explicit_camera_path_despite_owned_image_copies(live_service):
    service, camera, clock, inferred = live_service
    image = housing()
    previews = preview_frames(service, camera, clock, [image] * 3)
    assert all(preview["image"] is not image for preview in previews)
    assert len({id(preview["image"]) for preview in previews}) == 3
    result = service.inspect_live("inner_face", "physical-001")
    assert result["verdict"] == "PASS"
    assert result["metrics"]["capture_type"] == "live stable-frame selection"
    assert len(inferred) == 1
    assert inferred[0][1] == "inner_face"
    assert np.array_equal(inferred[0][0], image)
    assert service.session.missing_views == ["ribbed_face"]
    assert not service.session.counted


def test_repeated_latest_copies_of_one_sequence_do_not_establish_stability(live_service):
    service, camera, clock, inferred = live_service
    camera.packet = FramePacket(housing(), 1, 100.1)
    clock.now = 100.2
    for _ in range(4):
        service.preview()
    assert len(service._recent) == 1
    assert service.inspect_live("inner_face", "physical-001")["verdict"] == "RECHECK"
    assert_no_coverage(service, inferred)


def test_live_disconnected_camera_cannot_reuse_same_size_cached_frames(live_service):
    service, camera, clock, inferred = live_service
    image = housing()
    preview_frames(service, camera, clock, [image] * 3)
    camera.connected = False
    # The defensive mock still returns an image; connection diagnostics prevail.
    with pytest.raises(RuntimeError, match="disconnected"):
        service.inspect_live("inner_face", "physical-001")
    assert_no_coverage(service, inferred)


def test_stale_live_history_cannot_be_treated_as_an_imported_still(live_service):
    service, camera, clock, inferred = live_service
    preview_frames(service, camera, clock, [housing()] * 3)
    clock.now += 2
    result = service.inspect_live("inner_face", "physical-001")
    assert result["verdict"] == "RECHECK"
    assert_no_coverage(service, inferred)


def test_stale_current_packet_cannot_select_other_fresh_cached_frames(live_service):
    service, camera, clock, inferred = live_service
    image = housing()
    preview_frames(service, camera, clock, [image] * 3)
    # Provider inconsistency must fail closed instead of laundering an old
    # latest frame through a same-resolution cache from another capture.
    camera.packet = FramePacket(image.copy(), 4, 90.)
    result = service.inspect_live("inner_face", "physical-001")
    assert result["verdict"] == "RECHECK"
    assert_no_coverage(service, inferred)


def test_two_good_frames_are_insufficient_and_do_not_record_a_view(live_service):
    service, camera, clock, inferred = live_service
    preview_frames(service, camera, clock, [housing()] * 2)
    assert service.inspect_live("inner_face", "physical-001")["verdict"] == "RECHECK"
    assert_no_coverage(service, inferred)


def test_live_blurred_capture_is_recheck_without_recorded_coverage(live_service):
    service, camera, clock, inferred = live_service
    blurred = cv2.GaussianBlur(housing(), (25, 25), 6)
    preview_frames(service, camera, clock, [blurred] * 3)
    assert service.inspect_live("inner_face", "physical-001")["verdict"] == "RECHECK"
    assert_no_coverage(service, inferred)


def test_live_moving_part_is_recheck_without_recorded_coverage(live_service):
    service, camera, clock, inferred = live_service
    image = housing()
    moved = [np.roll(image, offset, axis=1) for offset in (0, 3, 6)]
    preview_frames(service, camera, clock, moved)
    assert service.inspect_live("inner_face", "physical-001")["verdict"] == "RECHECK"
    assert_no_coverage(service, inferred)


def test_live_frame_without_a_housing_cannot_record_inspection_coverage(live_service):
    service, camera, clock, inferred = live_service
    empty = np.full_like(housing(), (155, 95, 25))
    preview_frames(service, camera, clock, [empty] * 3)
    assert service.inspect_live("inner_face", "physical-001")["verdict"] == "RECHECK"
    assert_no_coverage(service, inferred)


def test_successful_analysis_resets_recent_frames_before_opposite_view(live_service):
    service, camera, clock, inferred = live_service
    image = housing()
    preview_frames(service, camera, clock, [image] * 3)
    service.inspect_live("inner_face", "physical-001")
    assert not service._recent
    assert service._capture_after == clock.now
    # Capturing again immediately must collect a new stable run after the flip.
    assert service.inspect_live("ribbed_face", "physical-001")["verdict"] == "RECHECK"
    assert len(inferred) == 1 and service.session.missing_views == ["ribbed_face"]
    preview_frames(service, camera, clock, [image] * 3, first_sequence=4,
                   timestamps=[100.5, 100.6, 100.7])
    assert service.inspect_live("ribbed_face", "physical-001")["verdict"] == "PASS"
    assert len(inferred) == 2 and service.session.complete
    assert service.session.preview().verdict == "PASS" and not service.session.counted


def test_stopping_camera_clears_capture_history_and_prevents_coverage(live_service):
    service, camera, clock, inferred = live_service
    preview_frames(service, camera, clock, [housing()] * 3)
    service.stop_camera()
    assert not service._recent
    with pytest.raises(RuntimeError, match="disconnected"):
        service.inspect_live("inner_face", "physical-001")
    assert_no_coverage(service, inferred)


def test_frames_before_begin_part_do_not_contribute_to_current_stability(live_service):
    service, camera, clock, inferred = live_service
    # Simulate a pre-session cached preview; lower-bound gating must exclude it.
    camera.packet = FramePacket(housing(), 1, 99.9)
    service._recent.extend([FramePacket(housing(), index, 99.6 + .1 * index) for index in range(3)])
    clock.now = 100.1
    assert service.inspect_live("inner_face", "physical-001")["verdict"] == "RECHECK"
    assert_no_coverage(service, inferred)
