"""The same native-coordinate inspection path for live captures and blind evaluation."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any
import cv2
import numpy as np

from src.acquisition.quality import QualityChecker
from src.artifacts import ArtifactStore
from src.inspection.fusion import EvidenceDecisionEngine
from src.inspection.session import ViewDecision
from src.utils.image_utils import read_image
from src.vision import GeometryProfile, HousingLocator, LandmarkRegistrar, roi_mask, tile_image, merge_tile_maps


@dataclass
class Analysis:
    valid: bool = False
    reasons: list[str] = field(default_factory=list)
    quality: Any = None
    registration: Any = None
    geometry: Any = None
    anomaly_map: np.ndarray | None = None
    anomaly_score: float | None = None
    mask: np.ndarray | None = None
    timings: dict = field(default_factory=dict)


def map_candidates(candidates: list[dict], inverse: np.ndarray, shape: tuple, view: str) -> list[dict]:
    mapped = []
    height, width = shape[:2]
    for item in candidates:
        points = np.asarray(item.get('contour', []), np.float32).reshape(-1, 1, 2)
        if len(points) < 3:
            continue
        points = cv2.perspectiveTransform(points, inverse).reshape(-1, 2)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        contour = np.rint(points).astype(np.int32)
        x, y, w, h = cv2.boundingRect(contour)
        mapped.append({**item, 'view': view, 'camera': 'top', 'contour': contour.tolist(),
                       'bbox': [x, y, w, h], 'bounding_box': [x, y, w, h],
                       'area_pixels': float(cv2.contourArea(contour)), 'coordinate_system': 'source_pixels'})
    return mapped


def render_candidates(image: np.ndarray, candidates: list[dict]) -> np.ndarray:
    output = image.copy()
    thickness = max(1, round(max(image.shape[:2]) / 1400))
    for candidate in candidates:
        contour = np.asarray(candidate.get('contour', []), np.int32).reshape(-1, 1, 2)
        if len(contour) >= 3:
            cv2.polylines(output, [contour], True, (0, 0, 255), thickness, cv2.LINE_AA)
    return output


class ViewPipeline:
    def __init__(self, directory: Path, mode: str = 'hybrid'):
        self.directory = Path(directory)
        self.metadata = ArtifactStore.verify(self.directory)
        self.view = self.metadata['view']
        self.mode = mode
        self.settings = self.metadata['settings']
        self.template = read_image(self.directory / 'template.png')
        self.registrar = LandmarkRegistrar(self.template, self.settings.get('registration'))
        self.geometry = GeometryProfile.load(self.directory / 'geometry.npz')
        self.rois = self.metadata.get('rois', [])
        self.thresholds = json.loads((self.directory / 'thresholds.json').read_text(encoding='utf-8'))
        self.quality = QualityChecker(self.settings.get('quality'))
        self.locator = HousingLocator()
        self.model = None
        if mode in {'hybrid', 'efficientad'}:
            from src.models.efficientad import EfficientADBackend
            self.model = EfficientADBackend(self.settings['models']['efficientad']).load(self.directory / 'neural')

    def analyze(self, frame: np.ndarray) -> Analysis:
        data = Analysis()
        started = time.perf_counter()
        expected = self.metadata.get('source_resolutions', [])
        if expected and [frame.shape[1], frame.shape[0]] not in expected:
            data.reasons = ['Input resolution was not represented in training/calibration; recapture at the commissioned native resolution']
            return data
        pose = self.locator.detect(frame)
        if pose is None:
            data.reasons = ['No complete circular housing was located']
            return data
        data.quality = self.quality.check(frame, mask=pose.mask)
        data.timings['quality_ms'] = (time.perf_counter() - started) * 1000
        if not data.quality.valid or not pose.complete:
            data.reasons = list(data.quality.reasons) + ([] if pose.complete else ['Housing is clipped by the frame'])
            return data
        began = time.perf_counter()
        data.registration = self.registrar.register(frame)
        data.timings['registration_ms'] = (time.perf_counter() - began) * 1000
        if not data.registration.valid or data.registration.confidence < self.settings['registration']['min_confidence']:
            data.reasons = list(data.registration.reasons) or ['Insufficient landmark registration confidence']
            return data
        image = data.registration.image
        aligned_pose = self.locator.detect(image)
        if aligned_pose is None or not aligned_pose.complete:
            data.reasons = ['Registered housing could not be localized']
            return data
        data.mask = cv2.bitwise_and(aligned_pose.mask, roi_mask(image.shape, self.rois))
        if not np.any(data.mask):
            data.reasons = ['Inspection ROI contains no part pixels']
            return data
        began = time.perf_counter()
        data.geometry = self.geometry.inspect(image, exclusion_mask=cv2.bitwise_not(roi_mask(image.shape, self.rois)))
        data.timings['geometry_ms'] = (time.perf_counter() - began) * 1000
        if not data.geometry.valid and self.mode != 'efficientad':
            data.reasons = ['Geometric measurements could not be established']
            return data
        if self.model is not None:
            began = time.perf_counter()
            settings = self.settings['tiling']
            tiles = list(tile_image(image, settings['tile_size'], settings['overlap'], data.mask))
            batch = max(1, int(settings['batch_size']))
            maps = []
            for start in range(0, len(tiles), batch):
                maps.extend(self.model.predict([tile.image for tile in tiles[start:start + batch]]))
            data.anomaly_map = merge_tile_maps(tiles, maps, image.shape)
            data.anomaly_map[data.mask == 0] = 0
            # A raw maximum has the same fixed semantics across train/eval/live.
            data.anomaly_score = float(np.max(data.anomaly_map[data.mask > 0]))
            data.timings['neural_ms'] = (time.perf_counter() - began) * 1000
        data.valid = True
        data.timings['total_ms'] = (time.perf_counter() - started) * 1000
        return data

    def inspect(self, frame: np.ndarray) -> tuple[ViewDecision, np.ndarray, Analysis]:
        data = self.analyze(frame)
        if not data.valid:
            decision = ViewDecision(self.view, 'RECHECK', data.reasons,
                                    metrics={'timings': data.timings}, model_version=self.metadata['version'])
            return decision, frame.copy(), data
        anomaly = None
        calibration = self.thresholds['geometry']
        decision_settings = {
            'registration_min_confidence': self.settings['registration']['min_confidence'],
            'min_registration_confidence': self.settings['registration']['min_confidence'],
            'geometry_reject_threshold': calibration['reject_threshold'],
            'geometry_recheck_threshold': calibration['recheck_threshold'],
            'geometry_calibrated': True,
            'calibration_version': self.metadata['version'],
            'geometry_calibration_version': self.metadata['version'],
        }
        if self.model is not None:
            calibration = self.thresholds['anomaly']
            threshold = calibration['reject_threshold']
            binary = ((data.anomaly_map >= threshold) & (data.mask > 0)).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            minimum = self.settings['localization']['minimum_defect_area']
            candidates = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if cv2.contourArea(contour) < minimum or min(w, h) < self.settings['localization'].get('minimum_defect_width', 2):
                    continue
                local = np.zeros(binary.shape, np.uint8)
                cv2.drawContours(local, [contour], -1, 255, -1)
                score = float(np.max(data.anomaly_map[local > 0]))
                candidates.append({'kind': 'surface anomaly candidate', 'source': 'EfficientAD',
                                   'contour': contour.reshape(-1, 2).tolist(), 'bbox': [x, y, w, h],
                                   'score': score, 'calibrated': True,
                                   'independent_support': False, 'measurements': {'area_pixels': cv2.contourArea(contour)}})
            anomaly = {'available': True, 'calibrated': True, 'score': data.anomaly_score,
                       'threshold': threshold, 'reject_threshold': threshold,
                       'recheck_threshold': calibration['recheck_threshold'],
                       'localization_threshold': threshold, 'candidates': candidates,
                       'model_version': self.metadata['version'], 'calibration_version': self.metadata['version']}
        decision = EvidenceDecisionEngine().decide(self.view, True, data.registration.confidence,
                                                   data.geometry, anomaly, mode=self.mode,
                                                   settings=decision_settings)
        decision.model_version = self.metadata['version']
        decision.metrics.update({'timings': data.timings, 'quality': data.quality.metrics,
                                 'registration_confidence': data.registration.confidence,
                                 'geometry_score': float(data.geometry.score),
                                 'geometry': data.geometry.measurements,
                                 'anomaly_score': data.anomaly_score,
                                 'scope': 'geometry only; surface defects not assessed' if self.mode == 'geometry' else self.mode})
        decision.candidates = map_candidates(decision.candidates, data.registration.inverse_matrix, frame.shape, self.view)
        if decision.verdict == 'FAIL' and not decision.candidates:
            decision.verdict = 'RECHECK'
            decision.reasons.append('Defect coordinates could not be mapped to the original image; recapture required')
        # An unlocalized candidate remains RECHECK, never an invented full-part contour.
        return decision, render_candidates(frame, decision.candidates), data
