"""Validation-only operating points with an explicit GOOD-part reject constraint."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import math

import numpy as np


def wilson_interval(successes: int, trials: int, confidence: float = .95) -> list[float] | None:
    """Two-sided Wilson score interval; undefined populations remain undefined."""
    if trials == 0:
        return None
    if not 0 <= successes <= trials:
        raise ValueError("Invalid binomial counts")
    from statistics import NormalDist
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie between zero and one")
    z = NormalDist().inv_cdf(.5 + confidence / 2)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1-p) / trials + z*z / (4*trials*trials)) / denominator
    return [max(0., centre-half), min(1., centre+half)]


def _scores(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Scores must be a one-dimensional collection of finite values")
    return array


def _part_max(scores: np.ndarray, ids: Iterable[str] | None, name: str) -> tuple[np.ndarray, list[str]]:
    if ids is None:
        raise ValueError(f"Explicit independent physical part IDs are required for {name}; image counts are not independent part counts")
    ids = list(ids)
    if len(ids) != len(scores) or any(not isinstance(x, str) or not x.strip() for x in ids):
        raise ValueError(f"One nonempty physical part ID is required per {name} score")
    grouped: dict[str, float] = {}
    for part_id, score in zip(ids, scores):
        grouped[part_id] = max(grouped.get(part_id, -math.inf), float(score))
    keys = sorted(grouped)
    return np.asarray([grouped[key] for key in keys]), keys


def _constrained_threshold(scores: np.ndarray, target: float) -> float:
    # Select the lowest raw-score threshold satisfying the empirical constraint.
    # Strictly above each observed value handles ties and inclusive >= semantics.
    candidates = np.r_[np.unique(scores), np.nextafter(np.unique(scores), np.inf)]
    for threshold in sorted(candidates):
        if np.isfinite(threshold) and np.count_nonzero(scores >= threshold) / len(scores) <= target + 1e-15:
            return float(threshold)
    raise ValueError("No finite threshold can be represented above the calibration scores")


@dataclass(frozen=True)
class OperatingPoint:
    reject_threshold: float
    recheck_threshold: float
    report: dict[str, Any]

    def decide(self, score: float) -> str:
        if not np.isfinite(score):
            return "RECHECK"
        if score >= self.reject_threshold:
            return "FAIL"
        if score >= self.recheck_threshold:
            return "RECHECK"
        return "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {"reject_threshold": self.reject_threshold,
                "recheck_threshold": self.recheck_threshold, "report": self.report}


def calibrate_operating_point(good_scores: Iterable[float], *,
                              good_part_ids: Iterable[str] | None = None,
                              target_fpr: float = .02, recheck_fpr: float = .1,
                              ng_scores: Iterable[float] = (),
                              ng_part_ids: Iterable[str] | None = None,
                              training_part_ids: Iterable[str] = (),
                              test_part_ids: Iterable[str] = (),
                              minimum_reject_score: float = 0.0,
                              minimum_recheck_score: float = 0.0) -> OperatingPoint:
    """Fit fixed raw-score limits using maxima per independent GOOD part.

    The constraint is empirical on validation data, never an accuracy guarantee.
    NG data describe the resulting trade-off and never relax the GOOD constraint.
    RECHECK is an ambiguity band, not a failure or a successful detection.
    """
    if not 0 <= target_fpr <= recheck_fpr < 1:
        raise ValueError("Require 0 <= target_fpr <= recheck_fpr < 1")
    good = _scores(good_scores)
    if not len(good):
        raise ValueError("Independent GOOD validation scores are required")
    good, good_ids = _part_max(good, good_part_ids, "GOOD validation")
    ng = _scores(ng_scores)
    ng_ids: list[str] = []
    if len(ng):
        ng, ng_ids = _part_max(ng, ng_part_ids, "NG validation")
    ids = set(good_ids) | set(ng_ids)
    if set(good_ids) & set(ng_ids):
        raise ValueError("A calibration physical part cannot have both GOOD and NG ground truth")
    if ids & (set(training_part_ids) | set(test_part_ids)):
        raise ValueError("Calibration part IDs overlap training or untouched test parts")
    if not np.isfinite([minimum_reject_score, minimum_recheck_score]).all() or min(minimum_reject_score, minimum_recheck_score) < 0:
        raise ValueError('Measurement score floors must be finite and nonnegative')
    reject = max(minimum_reject_score, minimum_recheck_score, _constrained_threshold(good, target_fpr))
    recheck = min(reject, max(minimum_recheck_score, _constrained_threshold(good, recheck_fpr)))
    fp = int(np.count_nonzero(good >= reject))
    good_ambiguous = int(np.count_nonzero((good >= recheck) & (good < reject)))
    tp = int(np.count_nonzero(ng >= reject))
    fn = int(np.count_nonzero(ng < recheck))
    ng_ambiguous = int(len(ng) - tp - fn)
    curve = []
    for threshold in np.unique(np.r_[good, ng, reject, recheck]):
        rejected_good = int(np.count_nonzero(good >= threshold))
        caught_ng = int(np.count_nonzero(ng >= threshold))
        curve.append({"threshold": float(threshold),
                      "false_positive_rate": rejected_good / len(good),
                      "defect_recall": caught_ng / len(ng) if len(ng) else None})
    interval = wilson_interval(fp, len(good))
    report: dict[str, Any] = {
        "schema_version": 1, "score_semantics": "raw score; greater is more anomalous; >= threshold rejects",
        "calibration_unit": "independent physical part; maximum score across its captures",
        "target_false_positive_rate": target_fpr, "target_nonpass_rate": recheck_fpr,
        "independent_good_parts": len(good), "independent_ng_parts": len(ng),
        "measurement_floor": {"reject": minimum_reject_score, "recheck": minimum_recheck_score},
        "good_part_ids": good_ids, "ng_part_ids": ng_ids,
        "observed_false_positive_rate": fp / len(good),
        "false_positive_rate_95ci": interval,
        "target_supported_by_95ci": bool(interval is not None and interval[1] <= target_fpr),
        "good_recheck_rate": good_ambiguous / len(good),
        "defect_recall": tp / len(ng) if len(ng) else None,
        "false_negative_rate": fn / len(ng) if len(ng) else None,
        "ng_recheck_rate": ng_ambiguous / len(ng) if len(ng) else None,
        "defect_recall_95ci": wilson_interval(tp, len(ng)),
        "ng_sensitivity_validated": bool(len(ng)), "operating_curve": curve,
        "warnings": (["No NG validation parts: missed-defect sensitivity is unknown"] if not len(ng) else [])
                    + (["Validation sample size does not support the target at 95% confidence"]
                       if interval is None or interval[1] > target_fpr else []),
    }
    return OperatingPoint(reject, recheck, report)


def calibrate_threshold(labels: Iterable[int], scores: Iterable[float], *,
                        part_ids: Iterable[str] | None = None,
                        target_fpr: float = .02, recheck_fpr: float = .1,
                        **kwargs: Any) -> tuple[float, dict[str, Any]]:
    """Convenience form with per-image labels; part ground truth uses any NG view."""
    labels, values = list(labels), _scores(scores)
    if len(labels) != len(values) or any(y not in (0, 1) for y in labels):
        raise ValueError("Binary labels and scores must have equal lengths")
    if part_ids is None:
        raise ValueError("Physical part IDs are required for leakage-safe calibration")
    ids = list(part_ids)
    if len(ids) != len(labels):
        raise ValueError("A physical part ID is required for every score")
    ng_parts = {p for p, y in zip(ids, labels) if y == 1}
    good_idx = [i for i, p in enumerate(ids) if p not in ng_parts]
    ng_idx = [i for i, p in enumerate(ids) if p in ng_parts]
    point = calibrate_operating_point(values[good_idx], good_part_ids=[ids[i] for i in good_idx],
                                     ng_scores=values[ng_idx], ng_part_ids=[ids[i] for i in ng_idx],
                                     target_fpr=target_fpr, recheck_fpr=recheck_fpr, **kwargs)
    return point.reject_threshold, {**point.report, "reject_threshold": point.reject_threshold,
                                    "recheck_threshold": point.recheck_threshold}


def localization_threshold(good_maps: Iterable[np.ndarray], quantile: float = .999) -> float:
    """One fixed map threshold; never normalize each image by its own maximum.

    This is a pixel operating point and requires separate part-level validation.
    """
    if not 0 < quantile < 1:
        raise ValueError("Localization quantile must lie between zero and one")
    arrays = [np.asarray(m, dtype=float).ravel() for m in good_maps]
    if not arrays or any(not len(m) or not np.isfinite(m).all() for m in arrays):
        raise ValueError("Finite independent GOOD validation maps are required")
    return float(np.quantile(np.concatenate(arrays), quantile))
