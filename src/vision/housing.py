"""Circular aluminium housing localization and conservative face identification."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PartPose:
    center: tuple[float, float]
    radius: float
    mask: np.ndarray
    complete: bool
    confidence: float = 0.0
    contour: np.ndarray | None = None


def hole_landmarks(image: np.ndarray, pose: PartPose, settings: dict | None = None,
                   keep: np.ndarray | None = None) -> list[dict]:
    """Visible dark circular openings, with local illumination compensation."""
    settings = settings or {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    local = cv2.GaussianBlur(gray, (0, 0), max(2., .035 * pose.radius)).astype(np.float32)
    dark = (local - gray > float(settings.get("hole_contrast", 20))).astype(np.uint8) * 255
    inside = np.zeros(gray.shape, np.uint8)
    cv2.circle(inside, tuple(round(v) for v in pose.center), round(pose.radius * .90), 255, -1)
    dark &= inside
    if keep is not None:
        dark &= keep
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not max(5, pose.radius ** 2 * .00015) <= area <= np.pi * (pose.radius * .15) ** 2:
            continue
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        if 4 * np.pi * area / max(perimeter * perimeter, 1) < .60 or min(w, h) / max(w, h) < .60:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if area / max(np.pi * radius ** 2, 1) >= .60:
            result.append({"center": [float(cx), float(cy)], "radius": float(radius)})
    return result


class HousingLocator:
    """Find a roughly circular foreground against a contrasting work surface.

    Segmentation uses robust border colour distance in Lab, so the photographed
    blue inspection mat and neutral backgrounds are supported. A touching image
    edge is explicitly recorded as an incomplete capture. This is localization,
    and deliberately provides no angular orientation estimate.
    """

    def __init__(self, settings: dict | None = None):
        self.settings = settings or {}

    def detect(self, image: np.ndarray) -> PartPose | None:
        if image is None or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            return None
        h, w = image.shape[:2]
        scale = min(1.0, 1000 / max(h, w))
        small = cv2.resize(image, (round(w * scale), round(h * scale))) if scale < 1 else image
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        border = np.concatenate([lab[:3].reshape(-1, 3), lab[-3:].reshape(-1, 3),
                                 lab[:, :3].reshape(-1, 3), lab[:, -3:].reshape(-1, 3)])
        background = np.median(border, axis=0)
        border_distance = np.linalg.norm(border - background, axis=1)
        mad = float(np.median(np.abs(border_distance - np.median(border_distance))))
        threshold = max(float(self.settings.get("foreground_lab_distance", 15)),
                        float(np.median(border_distance)) + 5 * 1.4826 * mad)
        distance = np.linalg.norm(lab - background, axis=2)
        binary = (distance > threshold).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            fraction = area / binary.size
            if not float(self.settings.get("min_area_ratio", .035)) <= fraction <= .92:
                continue
            perimeter = cv2.arcLength(contour, True)
            roundness = 4 * np.pi * area / max(perimeter ** 2, 1)
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            fill = area / max(np.pi * radius ** 2, 1)
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect = min(cw, ch) / max(cw, ch)
            if roundness < .40 or fill < .56 or aspect < .65:
                continue
            score = float((roundness * fill * aspect) ** (1 / 3) * min(1, fraction / .15))
            if score > best_score:
                best_score, best = score, (contour, cx, cy, radius)
        if best is None:
            return None
        contour, cx, cy, radius = best
        native_contour = np.rint(contour.astype(np.float64) / scale).astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [native_contour], -1, 255, -1)
        # Fill holes for the silhouette mask. Intentional holes are modelled separately.
        margin = max(2, .01 * radius / scale)
        points = native_contour[:, 0]
        complete = bool(points[:, 0].min() > margin and points[:, 1].min() > margin
                        and points[:, 0].max() < w - 1 - margin and points[:, 1].max() < h - 1 - margin)
        return PartPose((float(cx / scale), float(cy / scale)), float(radius / scale),
                        mask, complete, best_score, native_contour)


class FaceIdentifier:
    """A geometry cue, with an explicit unknown result for ambiguous surfaces.

    The operator-selected view remains useful when texture, glare, or a mounting
    fixture hides the radial ribs. These confidence values are heuristic cues,
    not calibrated classification probabilities.
    """

    def __init__(self, settings: dict | None = None):
        self.settings = settings or {}
        self.locator = HousingLocator(self.settings)

    def identify(self, image: np.ndarray) -> tuple[str | None, float]:
        pose = self.locator.detect(image)
        if pose is None or not pose.complete:
            return None, 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        polar = cv2.warpPolar(gray, (192, 720), pose.center, pose.radius,
                              cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS).astype(np.float32)
        band = polar[:, 65:173]
        # A radial fin produces the same angular edges at several radii.
        parts = np.array_split(band, 4, axis=1)
        profiles = [np.mean(p, axis=1) for p in parts]
        correlations = []
        for left, right in zip(profiles, profiles[1:]):
            if min(np.std(left), np.std(right)) > 2:
                correlations.append(float(np.corrcoef(left, right)[0, 1]))
        coherence = float(np.median(correlations)) if correlations else 0.0
        angular = float(np.std(np.mean(band, axis=1)))
        total = max(float(np.std(band)), 1)
        repeated = (profiles[1] - np.mean(profiles[1]))
        spectrum = np.abs(np.fft.rfft(repeated))
        rib_energy = float(np.sum(spectrum[10:120] ** 2) / max(np.sum(spectrum[1:] ** 2), 1))
        if coherence > .50 and angular > 3.5 and angular / total > .30 and rib_energy > .35:
            return "ribbed_face", float(np.clip(.50 + .30 * coherence + .20 * rib_energy, 0, .96))
        radial = np.mean(polar, axis=0)
        ring_contrast = float(np.percentile(radial[75:185], 95) - np.percentile(radial[75:185], 5))
        if coherence < .55 and ring_contrast > 18 and rib_energy < .35:
            return "inner_face", float(np.clip(.55 + ring_contrast / 300 - max(coherence, 0) / 8, .55, .88))
        return None, float(np.clip(max(coherence, 0) * .4, 0, .45))
