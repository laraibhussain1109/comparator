"""Learned, native-coordinate housing geometry inspection.

This is deterministic visual geometry, not calibrated dimensional metrology.
Inputs to ``fit`` and ``inspect`` must already share a confidently registered
pose. GOOD profiles learn intentional rib interruptions rather than assuming a
rib count. Independent validation is still required to set rejection thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os
from uuid import uuid4

import cv2
import numpy as np

from .housing import HousingLocator, PartPose, hole_landmarks
from .roi import roi_mask
from src.utils.filesystem import filesystem_path


@dataclass
class GeometryResult:
    score: float = 0.0
    candidates: list[dict] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)
    valid: bool = False


def _mask_at(masks: np.ndarray | list[np.ndarray] | None, index: int,
             shape: tuple[int, int], default: int) -> np.ndarray:
    if masks is None:
        return np.full(shape, default, np.uint8)
    value = masks[index] if isinstance(masks, (list, tuple)) else masks
    value = np.asarray(value)
    if value.shape != shape:
        raise ValueError("Geometry masks must match the native image dimensions")
    return (value != 0).astype(np.uint8) * 255


def _angle_blur(array: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(array, ((radius, radius), (0, 0)), mode="wrap")
    return cv2.blur(padded, (1, radius * 2 + 1))[radius:-radius]


class GeometryProfile:
    """Learned rib continuity, rim shape and repeatable visible holes.

    ``masks`` are keep masks, ``exclusions`` are nonzero keep-out masks; either
    may be a single native-size mask or one per training image. Definitions in
    settings["rois"] use :func:`roi_mask`. ``load(path)`` is a classmethod that
    returns the complete fitted profile, with no pickle/checkpoint execution.

    Scores are ratios to configurable image-geometry tolerances. A score is not
    a defect probability. One GOOD example is allowed for a research master,
    and its lack of manufacturing variation is recorded in measurements.
    """

    VERSION = 1

    def __init__(self, view: str, settings: dict | None = None):
        if view not in ("inner_face", "ribbed_face", "side_left", "side_right"):
            raise ValueError(f"Unsupported housing view: {view}")
        self.view = view
        self.settings = settings or {}
        self.locator = HousingLocator(self.settings)
        self.shape: tuple[int, int] | None = None
        self.center: tuple[float, float] | None = None
        self.radius = 0.0
        self.angle_bins = 0
        self.radial_bins = 0
        self.training_count = 0
        self.keep_mask: np.ndarray | None = None
        self.allowed_mask: np.ndarray | None = None
        self.structure: np.ndarray | None = None
        self.structure_mad: np.ndarray | None = None
        self.rim: np.ndarray | None = None
        self.rim_mad: np.ndarray | None = None
        self.holes: list[dict] = []
        self.circle_statistics: dict = {}
        self.mounting_edges: np.ndarray | None = None

    @property
    def max_radius(self) -> float:
        return self.radius * 1.08

    def _polar(self, image: np.ndarray, nearest: bool = False) -> np.ndarray:
        flags = cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS
        flags |= cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.warpPolar(image, (self.radial_bins, self.angle_bins),
                             self.center, self.max_radius, flags)

    def _cartesian(self, polar: np.ndarray) -> np.ndarray:
        return cv2.warpPolar(polar, (self.shape[1], self.shape[0]), self.center,
                             self.max_radius, cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP
                             | cv2.WARP_FILL_OUTLIERS | cv2.INTER_NEAREST)

    def _normalized_gray(self, image: np.ndarray, keep: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        values = gray[keep != 0]
        if values.size < 30:
            raise ValueError("Too few unmasked housing pixels")
        median = float(np.median(values))
        contrast = max(float(np.percentile(values, 90) - np.percentile(values, 10)), 25.)
        return (gray - median) / contrast

    def _rib_structure(self, gray: np.ndarray) -> np.ndarray:
        polar = self._polar(gray)
        smooth = cv2.GaussianBlur(np.pad(polar, ((3, 3), (0, 0)), mode="wrap"), (3, 3), .8)[3:-3]
        # Angular derivatives trace the long edges of radial fins. Keep the
        # radius dimension: a projection alone would hide chipped sections.
        derivative = .5 * (np.roll(smooth, -1, axis=0) - np.roll(smooth, 1, axis=0))
        return cv2.blur(np.abs(derivative), (5, 1)).astype(np.float32)

    def _rim_profile(self, mask: np.ndarray) -> np.ndarray:
        polar = self._polar(mask, nearest=True) != 0
        positions = np.arange(self.radial_bins, dtype=np.float32)[None, :]
        return np.max(np.where(polar, positions, 0), axis=1) * self.max_radius / self.radial_bins

    def _holes(self, image: np.ndarray, pose: PartPose, keep: np.ndarray) -> list[dict]:
        return hole_landmarks(image, pose, self.settings, keep)

    def _inner_circle(self, gray: np.ndarray) -> dict:
        """Fit the strongest recessed circular transition without inventing a ring."""
        polar = self._polar(gray)
        smoothed = cv2.GaussianBlur(polar, (5, 3), 1)
        derivative = np.abs(np.diff(smoothed, axis=1))
        projection = np.median(derivative, axis=0)
        low = round(.55 * self.radius / self.max_radius * self.radial_bins)
        high = round(.91 * self.radius / self.max_radius * self.radial_bins)
        peak = low + int(np.argmax(projection[low:high]))
        if float(projection[peak]) < .025:
            return {"detected": False}
        search = max(3, round(.025 * self.radial_bins))
        left, right = max(low, peak - search), min(high, peak + search + 1)
        radii = (left + np.argmax(derivative[:, left:right], axis=1)) * self.max_radius / self.radial_bins
        strengths = np.max(derivative[:, left:right], axis=1)
        selected = strengths > max(.025, .4 * float(projection[peak]))
        angles = np.arange(self.angle_bins) * 2 * np.pi / self.angle_bins
        points = np.column_stack([self.center[0] + radii * np.cos(angles),
                                  self.center[1] + radii * np.sin(angles)])[selected]
        if len(points) < self.angle_bins * .45:
            return {"detected": False}
        keep = np.ones(len(points), bool)
        for _ in range(3):
            active = points[keep]
            matrix = np.column_stack([2 * active[:, 0], 2 * active[:, 1], np.ones(len(active))])
            values = np.sum(active ** 2, axis=1)
            solution = np.linalg.lstsq(matrix, values, rcond=None)[0]
            cx, cy = solution[:2]
            fitted_radius = np.sqrt(max(0, solution[2] + cx * cx + cy * cy))
            residual = np.abs(np.linalg.norm(points - [cx, cy], axis=1) - fitted_radius)
            keep = residual < max(2., .015 * self.radius)
            if keep.sum() < self.angle_bins * .40:
                return {"detected": False}
        return {"detected": True, "center": [float(cx), float(cy)], "radius_px": float(fitted_radius),
                "concentricity_px": float(np.linalg.norm(np.array([cx, cy]) - self.center)),
                "median_fit_residual_px": float(np.median(residual[keep]))}

    def fit(self, images: list[np.ndarray], masks: np.ndarray | list[np.ndarray] | None = None,
            exclusions: np.ndarray | list[np.ndarray] | None = None) -> dict:
        if not images:
            raise ValueError("At least one registered GOOD housing image is required")
        self.shape = tuple(images[0].shape[:2])
        if any(image.shape[:2] != self.shape for image in images):
            raise ValueError("All GOOD geometry images must share the native registered dimensions")
        poses = [self.locator.detect(image) for image in images]
        if any(pose is None or not pose.complete for pose in poses):
            raise ValueError("Each GOOD geometry image must contain one complete localized housing")
        self.center = tuple(float(v) for v in np.median([pose.center for pose in poses], axis=0))
        self.radius = float(np.median([pose.radius for pose in poses]))
        centers = np.asarray([pose.center for pose in poses])
        if np.max(np.linalg.norm(centers - self.center, axis=1)) > .04 * self.radius:
            raise ValueError("GOOD images are not consistently registered")
        self.angle_bins = int(np.clip(np.ceil(2 * np.pi * self.radius), 720, 4096))
        self.radial_bins = max(64, int(np.ceil(self.max_radius)))
        self.training_count = len(images)
        keeps, allowed_masks, structures, rims, all_holes, circles, edges = [], [], [], [], [], [], []
        configured = roi_mask(self.shape, self.settings.get("rois", []))
        for index, (image, pose) in enumerate(zip(images, poses)):
            keep = _mask_at(masks, index, self.shape, 255) & configured
            keep[_mask_at(exclusions, index, self.shape, 0) != 0] = 0
            allowed_masks.append(keep.copy())
            keep &= pose.mask
            keeps.append(keep)
            gray = self._normalized_gray(image, keep)
            structures.append(self._rib_structure(gray))
            rims.append(self._rim_profile(pose.mask))
            all_holes.append(self._holes(image, pose, keep))
            if self.view == "inner_face":
                circles.append(self._inner_circle(gray))
            edges.append(cv2.Canny(np.clip(gray * 75 + 128, 0, 255).astype(np.uint8), 45, 110))
        self.keep_mask = np.minimum.reduce(keeps)
        self.allowed_mask = np.minimum.reduce(allowed_masks)
        self.structure = np.median(structures, axis=0).astype(np.float32)
        self.structure_mad = np.median(np.abs(np.asarray(structures) - self.structure), axis=0).astype(np.float32)
        self.rim = np.median(rims, axis=0).astype(np.float32)
        self.rim_mad = np.median(np.abs(np.asarray(rims) - self.rim), axis=0).astype(np.float32)
        self.mounting_edges = (np.mean(np.asarray(edges) > 0, axis=0) >= .8).astype(np.uint8) * 255
        # Cluster recurring holes around the first master's physical locations.
        # A feature must recur in >=70% of GOOD captures to become mandatory.
        self.holes = []
        for reference in all_holes[0]:
            observations = []
            for holes in all_holes:
                if not holes:
                    continue
                distances = [np.linalg.norm(np.array(h["center"]) - reference["center"]) for h in holes]
                best = int(np.argmin(distances))
                if distances[best] < max(4., .025 * self.radius):
                    observations.append(holes[best])
            if len(observations) >= np.ceil(.7 * len(images)):
                centers = np.array([h["center"] for h in observations])
                radii = np.array([h["radius"] for h in observations])
                self.holes.append({"center": np.median(centers, axis=0).tolist(),
                                   "radius": float(np.median(radii)),
                                   "position_tolerance_px": max(3., float(np.max(np.linalg.norm(centers - np.median(centers, axis=0), axis=1))) + 2.),
                                   "radius_tolerance_px": max(2., float(np.max(np.abs(radii - np.median(radii)))) + 1.5)})
        detected = [c for c in circles if c.get("detected")]
        self.circle_statistics = {}
        if len(detected) >= max(1, np.ceil(.7 * len(images))):
            for key in ("radius_px", "concentricity_px"):
                values = np.array([c[key] for c in detected])
                self.circle_statistics[key] = {"median": float(np.median(values)),
                                                "tolerance": max(3., float(np.max(np.abs(values - np.median(values)))) + 2.)}
        return {"view": self.view, "good_captures": len(images), "learned_holes": len(self.holes),
                "angle_bins": self.angle_bins, "radial_bins": self.radial_bins,
                "units": "pixels; normalized tolerance ratios", "requires_independent_calibration": True}

    def _candidate(self, kind: str, binary: np.ndarray, score: float, measurements: dict,
                   min_area: float = 8) -> list[dict]:
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            result.append({"kind": kind, "contour": contour[:, 0].tolist(), "bbox": [x, y, w, h],
                           "score": float(score), "source": "GEOMETRY", "measurements": measurements})
        return result

    def inspect(self, image: np.ndarray, exclusion_mask: np.ndarray | None = None) -> GeometryResult:
        if self.structure is None or self.shape is None:
            return GeometryResult(measurements={"error": "Geometry profile has not been fitted"})
        if image is None or image.shape[:2] != self.shape:
            return GeometryResult(measurements={"error": "Geometry input dimensions differ from the native master"})
        pose = self.locator.detect(image)
        if pose is None or not pose.complete:
            return GeometryResult(measurements={"error": "Housing is missing or incomplete"})
        center_error = float(np.linalg.norm(np.array(pose.center) - self.center))
        if center_error > max(4., .04 * self.radius) or abs(pose.radius / self.radius - 1) > .12:
            return GeometryResult(measurements={"error": "Housing geometry is not aligned with the GOOD profile",
                                                "center_error_px": center_error})
        keep = self.keep_mask.copy()
        keep[_mask_at(exclusion_mask, 0, self.shape, 0) != 0] = 0
        if np.count_nonzero(keep) < max(30, .02 * np.pi * self.radius ** 2):
            return GeometryResult(measurements={"error": "Inspection ROI contains too little housing area"})
        gray = self._normalized_gray(image, keep)
        measurements = {"units": "pixels and normalized image ratios", "training_good_captures": self.training_count,
                        "requires_independent_calibration": True, "outer_radius_px": pose.radius,
                        "center_error_px": center_error, "hole_count": 0, "rib_count_hardcoded": False}
        candidates: list[dict] = []
        polar_keep = self._polar(keep, nearest=True) != 0
        if self.view == "ribbed_face":
            current = self._rib_structure(gray)
            # Search a small neighbourhood before asserting loss of an edge.
            # This absorbs residual subpixel pose and normal casting variation.
            spatial_tolerance = float(self.settings.get("rib_position_tolerance_px", 2.0))
            angle_tolerance = max(1, round(spatial_tolerance * self.angle_bins / (2 * np.pi * self.radius * .65)))
            radial_tolerance = max(1, round(spatial_tolerance * self.radial_bins / self.max_radius))
            tolerant = cv2.dilate(np.pad(current, ((angle_tolerance, angle_tolerance), (0, 0)), mode="wrap"),
                                  np.ones((2 * angle_tolerance + 1, 2 * radial_tolerance + 1), np.uint8))[angle_tolerance:-angle_tolerance]
            tolerance = np.maximum(float(self.settings.get("rib_contrast_tolerance", .055)),
                                   float(self.settings.get("rib_mad_multiplier", 6)) * 1.4826 * self.structure_mad + .015)
            difference = self.structure - tolerant
            missing = ((difference > tolerance) & (tolerant < .45 * self.structure)
                       & (self.structure > float(self.settings.get("minimum_rib_edge", .065))))
            start = round(float(self.settings.get("rib_inner_radius", .33)) * self.radius / self.max_radius * self.radial_bins)
            stop = round(float(self.settings.get("rib_outer_radius", .96)) * self.radius / self.max_radius * self.radial_bins)
            annulus = np.zeros_like(missing)
            annulus[:, start:stop] = True
            missing &= annulus & polar_keep
            # Remove short texture fluctuations; retain long angular-edge tracks.
            radial_length = max(3, round(float(self.settings.get("minimum_rib_gap_radius", .06)) * self.radius / self.max_radius * self.radial_bins))
            binary = cv2.morphologyEx(missing.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                                      np.ones((1, radial_length), np.uint8))
            binary = cv2.dilate(binary, np.ones((max(3, 2 * angle_tolerance + 1), 3), np.uint8))
            native = self._cartesian(binary) & keep
            native = cv2.morphologyEx(native, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            nearby_expected = cv2.dilate(np.pad(self.structure, ((angle_tolerance, angle_tolerance), (0, 0)), mode="wrap"),
                                         np.ones((2 * angle_tolerance + 1, 2 * radial_tolerance + 1), np.uint8))[angle_tolerance:-angle_tolerance]
            excess = ((current - nearby_expected > tolerance) & (current > .065) & annulus & polar_keep)
            contours, _ = cv2.findContours(native, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < max(12, .00018 * np.pi * self.radius ** 2):
                    continue
                positions = contour[:, 0]
                radii = np.linalg.norm(positions - np.array(self.center), axis=1)
                span = float(np.ptp(radii))
                if span < float(self.settings.get("minimum_rib_gap_radius", .06)) * self.radius:
                    continue
                region = np.zeros(self.shape, np.uint8)
                cv2.drawContours(region, [contour], -1, 255, -1)
                selected = self._polar(region, nearest=True) != 0
                values = (difference / tolerance)[selected & missing]
                score = float(np.quantile(values, .8)) if values.size else 1.
                kind = "missing_rib" if span > .35 * self.radius else "interrupted_or_chipped_rib"
                search_angle = max(3, round(self.angle_bins * float(self.settings.get("bent_rib_search_degrees", 4)) / 360))
                nearby = cv2.dilate(selected.astype(np.uint8), np.ones((2 * search_angle + 1, 3), np.uint8)) != 0
                displacement_support = float(np.count_nonzero(excess & nearby) / max(1, np.count_nonzero(missing & selected)))
                if displacement_support > .35:
                    kind = "displaced_or_bent_rib"
                    # Include measured displaced edges, so the contour marks
                    # physical deformation as well as the lost expected track.
                    region |= self._cartesian((excess & nearby).astype(np.uint8) * 255)
                    region &= keep
                    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                candidates += self._candidate(kind, region, score,
                                               {"radial_gap_px": span, "radial_gap_ratio": span / self.radius,
                                                "displacement_edge_support_ratio": displacement_support})
            # A ridge moved beyond the search tolerance has a missing old edge;
            # unlike subtraction alone this does not require a global heatmap.
            measurements["rib_structure_coverage"] = float(np.mean(self.structure[:, start:stop] > .065))
        # Localized silhouette deviations, using fixed registered rays.
        current_rim = self._rim_profile(pose.mask)
        rim_tolerance = np.maximum(float(self.settings.get("rim_tolerance_px", 3.)), 6 * 1.4826 * self.rim_mad + 1.5)
        deviation = current_rim - self.rim
        measurements["rim_p95_deviation_px"] = float(np.quantile(np.abs(deviation), .95))
        bad_angles = np.abs(deviation) > rim_tolerance
        rim_polar = np.zeros((self.angle_bins, self.radial_bins), np.uint8)
        for angle in np.flatnonzero(bad_angles):
            lo = round(min(current_rim[angle], self.rim[angle]) / self.max_radius * self.radial_bins) - 2
            hi = round(max(current_rim[angle], self.rim[angle]) / self.max_radius * self.radial_bins) + 3
            rim_polar[angle, max(0, lo):min(self.radial_bins, hi)] = 255
        rim_native = self._cartesian(rim_polar)
        # The rim lies beyond the segmented mask at a chip, so use ROI keep-outs,
        # not the foreground-only mask for this detector.
        rim_allowed = self.allowed_mask.copy()
        rim_allowed[_mask_at(exclusion_mask, 0, self.shape, 0) != 0] = 0
        rim_native &= rim_allowed
        rim_native = cv2.morphologyEx(rim_native, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        if np.any(bad_angles):
            score = float(np.quantile(np.abs(deviation[bad_angles]) / rim_tolerance[bad_angles], .8))
            candidates += self._candidate("rim_shape_deviation", rim_native, score,
                                           {"max_deviation_px": float(np.max(np.abs(deviation)))},
                                           min_area=max(15, .00025 * np.pi * self.radius ** 2))
        observed_holes = self._holes(image, pose, keep)
        measurements["hole_count"] = len(observed_holes)
        measurements["expected_repeatable_holes"] = len(self.holes)
        used: set[int] = set()
        for expected in self.holes:
            ex, ey = expected["center"]
            if not keep[int(np.clip(round(ey), 0, self.shape[0] - 1)), int(np.clip(round(ex), 0, self.shape[1] - 1))]:
                continue
            distances = [np.linalg.norm(np.array(hole["center"]) - expected["center"]) if i not in used else np.inf
                         for i, hole in enumerate(observed_holes)]
            best = int(np.argmin(distances)) if distances else None
            search = max(expected["position_tolerance_px"] * 3, expected["radius"] * 1.5)
            missing_hole = best is None or distances[best] > search
            score, kind = 0., "hole_geometry_deviation"
            if missing_hole:
                score, kind = 2., "missing_hole"
            else:
                used.add(best)
                actual = observed_holes[best]
                score = float(max(distances[best] / expected["position_tolerance_px"],
                                  abs(actual["radius"] - expected["radius"]) / expected["radius_tolerance_px"]))
            if score > 1:
                local = np.zeros(self.shape, np.uint8)
                cv2.circle(local, (round(ex), round(ey)), max(4, round(expected["radius"] + 3)), 255, -1)
                local &= keep
                candidates += self._candidate(kind, local, score,
                                               {"expected_center_px": expected["center"],
                                                "expected_radius_px": expected["radius"]})
        if self.view == "inner_face":
            inner = self._inner_circle(gray)
            measurements["inner_circle"] = inner
            for key, baseline in self.circle_statistics.items():
                if inner.get("detected"):
                    measurements[f"inner_{key}_deviation"] = float(inner[key] - baseline["median"])
            # Ring measurements are diagnostic. Do not reject an entire surface
            # because a reflection made a fitted circular transition uncertain.
        if self.settings.get("check_mounting", False):
            regions = self.settings.get("mounting_regions", [])
            if regions:
                mounting_keep = roi_mask(self.shape, regions) & keep
                current_edges = cv2.Canny(np.clip(gray * 75 + 128, 0, 255).astype(np.uint8), 45, 110)
                distances = cv2.distanceTransform(255 - current_edges, cv2.DIST_L2, 3)
                tolerance = float(self.settings.get("mounting_edge_tolerance_px", 3.))
                missing_edges = ((self.mounting_edges != 0) & (distances > tolerance)
                                 & (mounting_keep != 0)).astype(np.uint8) * 255
                missing_edges = cv2.dilate(missing_edges, np.ones((3, 3), np.uint8))
                candidates += self._candidate("mounting_feature_deviation", missing_edges,
                                               float(np.max(distances[missing_edges != 0]) / tolerance) if np.any(missing_edges) else 0,
                                               {"edge_tolerance_px": tolerance}, min_area=max(15, self.radius * .15))
        score = max([float(c["score"]) for c in candidates], default=0.)
        measurements["candidate_count"] = len(candidates)
        return GeometryResult(score, candidates, measurements, True)

    def save(self, path: str | Path) -> None:
        if self.structure is None:
            raise ValueError("Cannot save an unfitted geometry profile")
        path = filesystem_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"version": self.VERSION, "view": self.view, "settings": self.settings,
                    "shape": self.shape, "center": self.center, "radius": self.radius,
                    "angle_bins": self.angle_bins, "radial_bins": self.radial_bins,
                    "training_count": self.training_count, "holes": self.holes,
                    "circle_statistics": self.circle_statistics}
        temporary = path.with_name(f".geometry-{uuid4().hex}.tmp")
        created = False
        try:
            with temporary.open("xb") as handle:
                created = True
                np.savez_compressed(handle, metadata=np.array(json.dumps(metadata)), keep_mask=self.keep_mask,
                                    allowed_mask=self.allowed_mask,
                                    structure=self.structure, structure_mad=self.structure_mad,
                                    rim=self.rim, rim_mad=self.rim_mad, mounting_edges=self.mounting_edges)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            if created:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> "GeometryProfile":
        with filesystem_path(path).open("rb") as handle, np.load(handle, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata["version"] != cls.VERSION:
                raise ValueError("Unsupported geometry profile version")
            profile = cls(metadata["view"], metadata["settings"])
            for key in ("radius", "angle_bins", "radial_bins", "training_count", "holes", "circle_statistics"):
                setattr(profile, key, metadata[key])
            profile.shape, profile.center = tuple(metadata["shape"]), tuple(metadata["center"])
            for key in ("keep_mask", "allowed_mask", "structure", "structure_mad", "rim", "rim_mad", "mounting_edges"):
                setattr(profile, key, archive[key].copy())
        return profile
