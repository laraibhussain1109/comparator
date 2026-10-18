"""Calibrated evidence rules; raw model scores are never averaged or voted."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

import cv2
import numpy as np

from src.inspection.session import ViewDecision


def _value(result: Any, key: str, default: Any = None) -> Any:
    return result.get(key, default) if isinstance(result, Mapping) else getattr(result, key, default)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _candidates(items: Any, source: str, threshold: float | None = None) -> list[dict[str, Any]]:
    """Accept measured spatial evidence, never a score-only fabricated box."""
    valid = []
    for item in items or []:
        if hasattr(item, "to_dict"):
            candidate = item.to_dict()
            if hasattr(item, "contour"):
                candidate["contour"] = np.asarray(item.contour).tolist()
        elif isinstance(item, Mapping):
            candidate = deepcopy(dict(item))
        else:
            continue
        contour = candidate.get("contour")
        bbox = candidate.get("bounding_box", candidate.get("bbox"))
        spatial = False
        if contour is not None:
            try:
                points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
                spatial = len(points) >= 3 and bool(np.isfinite(points).all()) and abs(cv2.contourArea(points.astype(np.float32))) > 0
            except (TypeError, ValueError, cv2.error):
                spatial = False
        if bbox is not None:
            try:
                values = np.asarray(bbox, dtype=np.float64).reshape(-1)
                spatial = spatial or (len(values) == 4 and bool(np.isfinite(values).all()) and
                                      values[0] >= 0 and values[1] >= 0 and values[2] > 0 and values[3] > 0)
            except (TypeError, ValueError):
                pass
        if not spatial:
            continue
        if threshold is not None:
            region_score = _finite(candidate.get("peak_score", candidate.get("score", candidate.get("raw_score"))))
            if region_score is None or region_score < threshold:
                continue
        candidate.setdefault("source", source)
        candidate.setdefault("type", "suspected structural deviation" if source == "geometry" else "surface anomaly")
        valid.append(candidate)
    return valid


def _map_candidates(anomaly_map: Any, threshold: float, minimum_area: float) -> list[dict[str, Any]]:
    if anomaly_map is None:
        return []
    values = np.asarray(anomaly_map)
    if values.ndim != 2 or not np.isfinite(values).all():
        return []
    mask = (values >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        if cv2.contourArea(contour) < minimum_area:
            continue
        region = np.zeros(values.shape, np.uint8)
        cv2.drawContours(region, [contour], -1, 1, -1)
        x, y, w, h = cv2.boundingRect(contour)
        result.append({"source": "anomaly", "type": "surface anomaly",
                       "bounding_box": [x, y, w, h], "contour": contour.tolist(),
                       "area_px": int(region.sum()), "peak_score": float(values[region > 0].max())})
    return result


class EvidenceDecisionEngine:
    def decide(self, view: str, quality_valid: bool, registration_confidence: float,
               geometry_result: Any, anomaly: dict[str, Any] | None,
               mode: str = "hybrid", settings: dict[str, Any] | None = None) -> ViewDecision:
        cfg = settings or {}
        aliases = {"geometry": "geometry_only", "geometry-only": "geometry_only",
                   "anomaly": "anomaly_only", "anomaly-only": "anomaly_only",
                   "efficientad": "anomaly_only", "dinomaly": "anomaly_only"}
        mode = aliases.get(mode, mode)
        if mode not in {"hybrid", "geometry_only", "anomaly_only"}:
            raise ValueError(f"Unknown inspection mode: {mode}")
        metrics: dict[str, Any] = {"mode": mode, "inspection_scope": mode,
                                   "registration_confidence": _finite(registration_confidence),
                                   "quality_valid": bool(quality_valid),
                                   "calibration_version": cfg.get("calibration_version")}
        version = anomaly.get("model_version") if anomaly else None

        def decision(verdict: str, reasons: list[str], candidates: list[dict] | None = None) -> ViewDecision:
            return ViewDecision(view, verdict, reasons, candidates or [], metrics, version)

        if not quality_valid:
            return decision("RECHECK", ["Frame quality or part stability is invalid; capture another frame"])
        registration_min = _finite(cfg.get("registration_min_confidence", cfg.get("min_registration_confidence", .6)))
        confidence = _finite(registration_confidence)
        if (registration_min is None or not 0 <= registration_min <= 1 or confidence is None or
                not 0 <= confidence <= 1 or confidence < registration_min):
            return decision("RECHECK", ["Landmark registration confidence is insufficient"])

        geometry_score = _finite(_value(geometry_result, "score"))
        geometry_reject = _finite(cfg.get("geometry_reject_threshold"))
        geometry_recheck = _finite(cfg.get("geometry_recheck_threshold"))
        geometry_valid = bool(_value(geometry_result, "valid", False))
        geometry_calibrated = (geometry_valid and geometry_score is not None and geometry_score >= 0 and
                               geometry_reject is not None and geometry_recheck is not None and
                               0 <= geometry_recheck <= geometry_reject and
                               cfg.get("geometry_calibrated", True) is not False)
        localized_geometry = _candidates(_value(geometry_result, "candidates", []), "geometry")
        metrics.update({"geometry_score": geometry_score,
                        "geometry_reject_threshold": geometry_reject,
                        "geometry_recheck_threshold": geometry_recheck,
                        "geometry_calibrated": geometry_calibrated,
                        "geometry_calibration_version": cfg.get("geometry_calibration_version"),
                        "geometry_measurements": deepcopy(_value(geometry_result, "measurements", {}))})

        if mode != "anomaly_only" and geometry_calibrated and geometry_score >= geometry_reject:
            if localized_geometry:
                return decision("FAIL", ["Calibrated geometry rejection threshold exceeded with localized structural evidence"], localized_geometry)
            return decision("RECHECK", ["Geometry deviation exceeds rejection threshold but cannot be localized"])

        unresolved = []
        if mode != "anomaly_only":
            if not geometry_calibrated:
                unresolved.append("Geometry profile or independent validation calibration is unavailable")
            elif geometry_score >= geometry_recheck:
                unresolved.append("Geometry deviation lies in the calibrated recheck range")
            elif localized_geometry:
                # A profile's candidate extraction threshold may be more sensitive
                # than its independent validation operating point. Do not mark it
                # red or reject it solely because a local feature was proposed.
                metrics["subthreshold_geometry_candidates"] = len(localized_geometry)

        if mode == "geometry_only":
            if unresolved:
                return decision("RECHECK", unresolved)
            return decision("PASS", ["Calibrated geometry checks passed; surface defects were not assessed"])

        if not anomaly or anomaly.get("available") is not True:
            return decision("RECHECK", unresolved + ["The selected neural model is unavailable; surface inspection was not completed"])
        image_score = _finite(anomaly.get("score"))
        reject = _finite(anomaly.get("threshold"))
        recheck = _finite(anomaly.get("recheck_threshold"))
        localization = _finite(anomaly.get("localization_threshold"))
        calibrated = (anomaly.get("calibrated") is True and image_score is not None and image_score >= 0 and
                      reject is not None and recheck is not None and localization is not None and
                      0 <= recheck <= reject and localization >= 0)
        metrics.update({"anomaly_score": image_score, "anomaly_reject_threshold": reject,
                        "anomaly_recheck_threshold": recheck,
                        "localization_threshold": localization, "anomaly_calibrated": calibrated,
                        "anomaly_calibration_version": anomaly.get("calibration_version")})
        if not calibrated:
            return decision("RECHECK", unresolved + ["Neural scores or spatial thresholds lack valid independent calibration"])
        candidates = _candidates(anomaly.get("candidates", []), "anomaly", localization)
        if not candidates:
            candidates = _map_candidates(anomaly.get("map", anomaly.get("anomaly_map")), localization,
                                          float(cfg.get("minimum_defect_area", 9)))
        if image_score >= reject:
            if candidates:
                return decision("FAIL", ["Calibrated surface-anomaly rejection threshold exceeded with localized evidence"], candidates)
            unresolved.append("High image anomaly score has no calibrated spatial evidence")
        elif image_score >= recheck:
            unresolved.append("Surface anomaly lies in the calibrated recheck range")
        elif candidates:
            unresolved.append("Localized surface candidate remains unresolved below the image rejection threshold")
        if unresolved:
            return decision("RECHECK", unresolved, candidates)
        reasons = ["Calibrated geometry and surface-anomaly checks passed"]
        if mode == "anomaly_only":
            reasons = ["Calibrated surface-anomaly checks passed; deterministic geometry was not assessed"]
        return decision("PASS", reasons)
