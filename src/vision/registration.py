"""Landmark-constrained registration; a circular silhouette never supplies angle."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .housing import HousingLocator, hole_landmarks


@dataclass
class Registration:
    image: np.ndarray
    confidence: float
    matrix: np.ndarray
    inverse_matrix: np.ndarray
    reasons: list[str] = field(default_factory=list)
    valid: bool = False

    def source_points(self, points: np.ndarray) -> np.ndarray:
        """Map aligned defect points back to the original captured image."""
        array = np.asarray(points, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(array, self.inverse_matrix).reshape(-1, 2)


class LandmarkRegistrar:
    def __init__(self, template: np.ndarray, settings: dict | None = None):
        if template is None or template.ndim != 3 or not template.size:
            raise ValueError("A native BGR template image is required")
        self.template = template.copy()
        self.settings = settings or {}
        border = np.concatenate([template[0], template[-1], template[:, 0], template[:, -1]])
        self._border_colour = tuple(float(v) for v in np.median(border, axis=0))
        self.locator = HousingLocator(self.settings)
        self.template_pose = self.locator.detect(template)
        self._template_holes = hole_landmarks(template, self.template_pose, self.settings) if self.template_pose else []
        self.orb = cv2.ORB_create(nfeatures=int(self.settings.get("orb_features", 6000)),
                                   scaleFactor=1.18, nlevels=8, edgeThreshold=15,
                                   patchSize=31, fastThreshold=10)
        self._template_points, self._template_descriptors = self._features(template)

    def _features(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        pose = self.locator.detect(image)
        if pose is None:
            return np.empty((0, 2), np.float32), None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Exterior rim and silhouette edges are excluded: they cannot determine angle.
        mask = np.zeros(gray.shape, np.uint8)
        cv2.circle(mask, tuple(round(v) for v in pose.center), round(.84 * pose.radius), 255, -1)
        scale = min(1.0, float(self.settings.get("feature_max_dimension", 1800)) / max(gray.shape))
        if scale < 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        gray = cv2.createCLAHE(2, (8, 8)).apply(gray)
        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)
        points = np.asarray([k.pt for k in keypoints], np.float32).reshape(-1, 2) / scale
        return points, descriptors

    def _failure(self, image: np.ndarray, reason: str, confidence: float = 0) -> Registration:
        fallback = self._hole_registration(image)
        if fallback is not None:
            return fallback
        # The returned image must never be interpreted as aligned when valid=False.
        h, w = self.template.shape[:2]
        canvas = np.zeros_like(self.template)
        ch, cw = min(h, image.shape[0]), min(w, image.shape[1])
        canvas[:ch, :cw] = image[:ch, :cw]
        return Registration(canvas, confidence, np.eye(3), np.eye(3), [reason], False)

    def _hole_registration(self, image: np.ndarray) -> Registration | None:
        """Match an asymmetric constellation when smooth metal defeats ORB.

        Four or more mutually consistent openings and a unique rotation are
        required. The circular rim only validates translation/scale afterwards.
        A regular bolt circle has several equivalent poses and is rejected.
        """
        pose, reference_pose = self.locator.detect(image), self.template_pose
        if pose is None or reference_pose is None or not pose.complete or not reference_pose.complete:
            return None
        moving = sorted(hole_landmarks(image, pose, self.settings), key=lambda h: -h["radius"])[:12]
        reference = sorted(self._template_holes, key=lambda h: -h["radius"])[:12]
        minimum = int(self.settings.get("minimum_hole_landmarks", 4))
        if min(len(moving), len(reference)) < minimum:
            return None
        source = np.array([h["center"] for h in moving], np.float64)
        target = np.array([h["center"] for h in reference], np.float64)
        sr = np.array([h["radius"] for h in moving])
        tr = np.array([h["radius"] for h in reference])
        expected_scale = reference_pose.radius / pose.radius
        tolerance = max(2.5, .012 * reference_pose.radius)
        reference_pairs = [(a, b, target[b] - target[a]) for a in range(len(target)) for b in range(len(target))
                           if a != b and np.linalg.norm(target[b] - target[a]) > .13 * reference_pose.radius]
        solutions: list[tuple[int, float, float, np.ndarray, list[tuple[int, int]]]] = []
        seen: set[tuple[tuple[int, int], ...]] = set()
        for a in range(len(source)):
            for b in range(len(source)):
                delta = source[b] - source[a]
                length = np.linalg.norm(delta)
                if a == b or length < .13 * pose.radius:
                    continue
                for ta, tb, target_delta in reference_pairs:
                    scale = np.linalg.norm(target_delta) / length
                    if abs(scale / expected_scale - 1) > .035:
                        continue
                    if max(abs(sr[a] * scale / tr[ta] - 1), abs(sr[b] * scale / tr[tb] - 1)) > .28:
                        continue
                    complex_ratio = complex(*target_delta) / complex(*delta)
                    real, imag = complex_ratio.real, complex_ratio.imag
                    affine = np.array([[real, -imag, 0.], [imag, real, 0.]])
                    affine[:, 2] = target[ta] - affine[:, :2] @ source[a]
                    transformed = source @ affine[:, :2].T + affine[:, 2]
                    distance = np.linalg.norm(transformed[:, None] - target[None], axis=2)
                    radius_compatible = np.abs(sr[:, None] * scale / tr[None] - 1) < .30
                    distance[~radius_compatible] = np.inf
                    pairs = []
                    used_source, used_target = set(), set()
                    for flat in np.argsort(distance, axis=None):
                        si, ti = np.unravel_index(flat, distance.shape)
                        if distance[si, ti] > tolerance:
                            break
                        if si not in used_source and ti not in used_target:
                            pairs.append((int(si), int(ti)))
                            used_source.add(si)
                            used_target.add(ti)
                    signature = tuple(sorted(pairs))
                    if len(pairs) < minimum or signature in seen:
                        continue
                    seen.add(signature)
                    matched_source = source[[s for s, t in pairs]].astype(np.float32)
                    matched_target = target[[t for s, t in pairs]].astype(np.float32)
                    refined, _ = cv2.estimateAffinePartial2D(matched_source, matched_target, method=cv2.LMEDS)
                    if refined is None:
                        continue
                    errors = np.linalg.norm(cv2.transform(matched_source[:, None], refined).reshape(-1, 2) - matched_target, axis=1)
                    if np.max(errors) > tolerance:
                        continue
                    error = float(np.mean(errors))
                    coverage = cv2.contourArea(cv2.convexHull(matched_target)) / (np.pi * reference_pose.radius ** 2)
                    if coverage < .01:
                        continue
                    angle = float(np.arctan2(refined[1, 0], refined[0, 0]))
                    solutions.append((len(pairs), error, angle, refined, pairs))
        if not solutions:
            return None
        solutions.sort(key=lambda s: (-s[0], s[1]))
        count, error, angle, affine, pairs = solutions[0]
        if count / min(len(source), len(target)) < .70:
            return None
        for other_count, other_error, other_angle, _, _ in solutions[1:]:
            angle_difference = abs(np.arctan2(np.sin(other_angle - angle), np.cos(other_angle - angle)))
            if other_count >= count and other_error <= error + .6 and angle_difference > np.deg2rad(5):
                return None
        scale = float(np.hypot(affine[0, 0], affine[1, 0]))
        mapped_center = affine[:, :2] @ pose.center + affine[:, 2]
        if (np.linalg.norm(mapped_center - reference_pose.center) > .035 * reference_pose.radius
                or abs(scale / expected_scale - 1) > .035):
            return None
        confidence = float(np.clip(.65 + .20 * count / max(len(source), len(target)) + .10 * (1 - error / tolerance), 0, .95))
        if confidence < float(self.settings.get("min_confidence", self.settings.get("registration_confidence", .60))):
            return None
        matrix = np.eye(3)
        matrix[:2] = affine
        aligned = cv2.warpAffine(image, affine, (self.template.shape[1], self.template.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderValue=self._border_colour)
        return Registration(aligned, confidence, matrix, np.linalg.inv(matrix),
                            [f"Unique asymmetric constellation of {count} hole landmarks; mean error {error:.2f} px"], True)

    def register(self, image: np.ndarray) -> Registration:
        if image is None or image.ndim != 3 or not image.size:
            raise ValueError("A nonempty BGR image is required")
        pose = self.locator.detect(image)
        if pose is None or self.template_pose is None:
            return self._failure(image, "Housing or template housing could not be localized")
        if not pose.complete or not self.template_pose.complete:
            return self._failure(image, "Housing boundary is clipped")
        points, descriptors = self._features(image)
        reference = self._template_descriptors
        minimum = int(self.settings.get("minimum_landmarks", 8))
        if descriptors is None or reference is None or min(len(descriptors), len(reference)) < minimum:
            return self._failure(image, "Insufficient internal landmarks; silhouette cannot establish orientation")
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        ratio = float(self.settings.get("match_ratio", .72))
        forward = matcher.knnMatch(descriptors, reference, k=2)
        reverse = matcher.knnMatch(reference, descriptors, k=2)
        reverse_good = {m.queryIdx: m.trainIdx for pair in reverse if len(pair) == 2
                        for m, n in [pair] if m.distance < ratio * n.distance}
        matches = [m for pair in forward if len(pair) == 2 for m, n in [pair]
                   if m.distance < ratio * n.distance and reverse_good.get(m.trainIdx) == m.queryIdx]
        if len(matches) < minimum:
            return self._failure(image, "Too few distinct mutually matched internal landmarks")
        source = np.float32([points[m.queryIdx] for m in matches])
        target = np.float32([self._template_points[m.trainIdx] for m in matches])
        tolerance = max(2., float(self.settings.get("ransac_error_radius", .008)) * self.template_pose.radius)
        affine, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                                      ransacReprojThreshold=tolerance, maxIters=4000,
                                                      confidence=.999, refineIters=15)
        if affine is None or inliers is None:
            return self._failure(image, "Internal landmark RANSAC did not converge")
        selected = inliers.ravel().astype(bool)
        count = int(selected.sum())
        fraction = count / len(matches)
        if count < minimum or fraction < float(self.settings.get("minimum_inlier_ratio", .45)):
            return self._failure(image, "Landmark consensus is insufficient", fraction * .4)
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2] = affine
        transformed = cv2.transform(source[selected, None], affine).reshape(-1, 2)
        error = float(np.median(np.linalg.norm(transformed - target[selected], axis=1)))
        span = cv2.contourArea(cv2.convexHull(target[selected])) / (np.pi * self.template_pose.radius ** 2)
        scale = float(np.hypot(affine[0, 0], affine[1, 0]))
        mapped_center = cv2.transform(np.float32([[pose.center]]), affine).reshape(2)
        center_error = np.linalg.norm(mapped_center - self.template_pose.center) / self.template_pose.radius
        radius_error = abs(scale * pose.radius / self.template_pose.radius - 1)
        if span < float(self.settings.get("minimum_landmark_coverage", .025)):
            return self._failure(image, "Internal landmarks occupy too small a region to validate pose")
        if not .5 <= scale <= 2 or center_error > .06 or radius_error > .08 or error > tolerance:
            return self._failure(image, "Estimated landmark transform disagrees with housing geometry")
        confidence = float(np.clip(.35 * fraction + .25 * min(1, count / 30)
                                    + .20 * min(1, span / .20)
                                    + .20 * max(0, 1 - error / tolerance), 0, 1))
        minimum_confidence = float(self.settings.get("min_confidence", self.settings.get("registration_confidence", .60)))
        if confidence < minimum_confidence:
            return self._failure(image, "Landmark registration confidence is below the configured minimum", confidence)
        aligned = cv2.warpAffine(image, affine, (self.template.shape[1], self.template.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=self._border_colour)
        return Registration(aligned, confidence, matrix, np.linalg.inv(matrix),
                            [f"{count}/{len(matches)} internal landmark inliers; median error {error:.2f} px"], True)
