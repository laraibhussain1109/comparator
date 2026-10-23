"""Conservative capture gates. Thresholds require commissioning on real parts.

Temporal stability here selects a clean capture; repeated video frames are not
independent evidence of a defect and must not be counted as such by fusion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Iterable

import cv2
import numpy as np

from .camera import FramePacket


@dataclass
class QualityReport:
    valid: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


DEFAULTS: dict[str, Any] = {
    "min_sharpness": 35.0,  # native-resolution Laplacian variance in intensity²
    "min_brightness": 25.0,
    "max_brightness": 235.0,
    "dark_level": 5,
    "bright_level": 250,
    "max_dark_fraction": .25,
    "max_bright_fraction": .20,
    "max_motion": .025,  # mean brightness-compensated difference / 255
    "motion_max_dimension": 640,
    "min_part_fraction": .01,
    "max_part_fraction": .95,
    "border_margin": 3,
    "require_mask": False,
    "min_pose_confidence": .45,
    "stable_frames": 3,
    "max_frame_age": 1.0,
    "max_stable_gap": .25,
    "selection_window": 5,
}


class QualityChecker:
    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self.settings = {**DEFAULTS, **(settings or {})}
        for name in ("max_dark_fraction", "max_bright_fraction", "max_motion", "min_part_fraction", "max_part_fraction"):
            if not 0 <= self.settings[name] <= 1:
                raise ValueError(f"{name} must be a fraction between 0 and 1")
        if (self.settings["stable_frames"] < 1 or self.settings["max_frame_age"] <= 0
                or self.settings["max_stable_gap"] <= 0 or self.settings["selection_window"] < 1):
            raise ValueError("Stable frame count and maximum age must be positive")

    def check(self, image: np.ndarray, mask: np.ndarray | None = None,
              previous: np.ndarray | None = None, *,
              pose_confidence: float | None = None) -> QualityReport:
        reasons: list[str] = []
        metrics: dict[str, Any] = {"motion_checked": False, "part_presence_checked": mask is not None}
        if not self._valid_image(image):
            return QualityReport(False, ["invalid_image"], metrics)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        height, width = gray.shape
        valid_mask = np.ones_like(gray, dtype=np.uint8)
        if mask is None:
            if self.settings["require_mask"]:
                reasons.append("part_mask_required")
        else:
            if not isinstance(mask, np.ndarray) or mask.shape != gray.shape or not np.isfinite(mask).all():
                return QualityReport(False, ["invalid_part_mask"], metrics)
            valid_mask = (mask > 0).astype(np.uint8)
            part_fraction = float(np.mean(valid_mask))
            metrics["part_fraction"] = part_fraction
            if part_fraction < self.settings["min_part_fraction"]:
                reasons.append("part_absent_or_too_small")
            if part_fraction > self.settings["max_part_fraction"]:
                reasons.append("part_too_large")
            if not np.any(valid_mask):
                return QualityReport(False, reasons or ["empty_part_mask"], metrics)
            margin = max(1, int(self.settings["border_margin"]))
            touches = (valid_mask[:margin].any() or valid_mask[-margin:].any()
                       or valid_mask[:, :margin].any() or valid_mask[:, -margin:].any())
            metrics["part_complete"] = not bool(touches)
            if touches:
                reasons.append("part_clipped_at_frame_edge")
        selected = gray[valid_mask.astype(bool)]
        brightness = float(np.mean(selected))
        dark = float(np.mean(selected <= self.settings["dark_level"]))
        bright = float(np.mean(selected >= self.settings["bright_level"]))
        # Exclude the silhouette and hole boundaries from the focus estimate.
        focus_mask = cv2.erode(valid_mask, np.ones((5, 5), np.uint8))
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        focus_pixels = laplacian[focus_mask.astype(bool)]
        sharpness = float(np.var(focus_pixels)) if focus_pixels.size else 0.0
        metrics.update(brightness=brightness, dark_fraction=dark, bright_fraction=bright,
                       sharpness=sharpness, resolution=[width, height])
        if brightness < self.settings["min_brightness"]:
            reasons.append("underexposed")
        if brightness > self.settings["max_brightness"]:
            reasons.append("overexposed")
        if dark > self.settings["max_dark_fraction"]:
            reasons.append("excessive_dark_clipping")
        if bright > self.settings["max_bright_fraction"]:
            reasons.append("excessive_highlight_clipping")
        if sharpness < self.settings["min_sharpness"]:
            reasons.append("insufficient_focus")
        if previous is not None:
            if not self._valid_image(previous) or previous.shape != image.shape:
                reasons.append("previous_frame_shape_mismatch")
            else:
                prior = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY) if previous.ndim == 3 else previous
                scale = min(1., self.settings["motion_max_dimension"] / max(height, width))
                size = (max(1, int(width * scale)), max(1, int(height * scale)))
                current_small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32)
                prior_small = cv2.resize(prior, size, interpolation=cv2.INTER_AREA).astype(np.float32)
                roi_small = cv2.resize(valid_mask, size, interpolation=cv2.INTER_NEAREST).astype(bool)
                delta = (current_small - prior_small)[roi_small]
                if delta.size:
                    motion = float(np.mean(np.abs(delta - np.median(delta))) / 255.)
                    metrics.update(motion=motion, motion_checked=True)
                    if motion > self.settings["max_motion"]:
                        reasons.append("part_moving")
                else:
                    reasons.append("empty_motion_roi")
        if pose_confidence is not None:
            metrics["pose_confidence"] = float(pose_confidence)
            if not np.isfinite(pose_confidence) or pose_confidence < self.settings["min_pose_confidence"]:
                reasons.append("uncertain_pose")
        return QualityReport(not reasons, reasons, metrics)

    @staticmethod
    def _valid_image(image: Any) -> bool:
        return (isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.size > 0
                and (image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3))
                and min(image.shape[:2]) >= 5)

    def select_sharpest(self, frames: Iterable[FramePacket], mask: np.ndarray | None = None,
                        *, now: float | None = None, min_stable_frames: int | None = None,
                        max_age: float | None = None) -> FramePacket | None:
        """Select an owned frame from the latest consecutive stable, valid run.

        Duplicate sequence IDs do not satisfy stability. Stale, future-dated,
        moving or invalid captures cannot be returned. This is a capture-quality
        operation, not temporal defect confirmation.
        """
        realtime = now is None
        now = time.monotonic() if realtime else now
        max_age = self.settings["max_frame_age"] if max_age is None else max_age
        required = self.settings["stable_frames"] if min_stable_frames is None else min_stable_frames
        if required < 1 or max_age <= 0:
            raise ValueError("Stable frame count and maximum age must be positive")
        unique = {frame.sequence: frame for frame in frames if 0 <= now - frame.timestamp <= max_age}
        ordered = sorted(unique.values(), key=lambda frame: frame.timestamp)
        # Native 4K focus checks are deliberately bounded. Checking an entire
        # preview history can make every candidate stale before selection ends.
        ordered = ordered[-max(required, int(self.settings["selection_window"])):]
        stable: list[tuple[float, FramePacket]] = []
        previous = None
        previous_sequence = None
        previous_timestamp = None
        for frame in ordered:
            # Dropped preview frames are expected; non-increasing sequence IDs
            # indicate stale/reordered input and cannot establish stability.
            if previous_sequence is not None and (frame.sequence <= previous_sequence
                    or frame.timestamp - previous_timestamp > self.settings["max_stable_gap"]):
                stable.clear()
                previous = None
            report = self.check(frame.image, mask, previous)
            if report.valid:
                stable.append((report.metrics["sharpness"], frame))
            else:
                stable.clear()
            previous = frame.image
            previous_sequence = frame.sequence
            previous_timestamp = frame.timestamp
        # Quality analysis itself takes measurable time on native frames. An
        # initially fresh candidate must still be fresh when returned.
        finished = time.monotonic() if realtime else now
        stable = [(score, frame) for score, frame in stable if 0 <= finished - frame.timestamp <= max_age]
        if len(stable) < required:
            return None
        best = max(stable, key=lambda item: (item[0], item[1].timestamp))[1]
        return FramePacket(best.image.copy(), best.sequence, best.timestamp)


def select_sharpest(frames: Iterable[FramePacket], checker: QualityChecker | None = None,
                    mask: np.ndarray | None = None, **kwargs: Any) -> FramePacket | None:
    return (checker or QualityChecker()).select_sharpest(frames, mask, **kwargs)
