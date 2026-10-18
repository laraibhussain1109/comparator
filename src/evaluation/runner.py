"""Blind evaluation calls the supplied production inspection callback unchanged."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable
import csv
import json
import math
import time

import cv2
import numpy as np

from src.data import DatasetIntegrityError, Sample
from .calibration import wilson_interval


_VERDICTS = {"PASS": "PASS", "FAIL": "FAIL", "RECHECK": "RECHECK", "GOOD": "PASS", "NG": "FAIL"}


def _get(result: Any, name: str, default: Any = None) -> Any:
    return result.get(name, default) if isinstance(result, dict) else getattr(result, name, default)


def _outcome(result: Any) -> str:
    value = _get(result, "verdict", _get(result, "result", _get(result, "status", "RECHECK")))
    value = getattr(value, "value", value)
    return _VERDICTS.get(str(value).upper(), "RECHECK")


def _score(result: Any, key: str) -> float | None:
    scores = _get(result, "scores", {})
    value = scores.get(key) if isinstance(scores, dict) else None
    metrics = _get(result, "metrics", {})
    if value is None and isinstance(metrics, dict):
        value = metrics.get(key)
    if value is None:
        value = _get(result, "anomaly_score", _get(result, "score", None))
    if value is not None:
        try:
            value = float(value)
            if math.isfinite(value):
                return value
        except (ValueError, TypeError):
            pass
    return None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def decision_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    tp = sum(r["label"] == 1 and r["verdict"] == "FAIL" for r in records)
    tn = sum(r["label"] == 0 and r["verdict"] == "PASS" for r in records)
    fp = sum(r["label"] == 0 and r["verdict"] == "FAIL" for r in records)
    fn = sum(r["label"] == 1 and r["verdict"] == "PASS" for r in records)
    good = sum(r["label"] == 0 for r in records)
    ng = sum(r["label"] == 1 for r in records)
    recheck = sum(r["verdict"] == "RECHECK" for r in records)
    decisive = len(records) - recheck
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else
          (0. if ng and decisive else None))
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]
    auc = None
    scored = [r for r in records if r.get("score") is not None]
    if len(scored) == len(records) and len({r["label"] for r in scored}) == 2:
        # Rank-statistic AUROC supports ties without a heavyweight ML dependency.
        positive = [r["score"] for r in scored if r["label"] == 1]
        negative = [r["score"] for r in scored if r["label"] == 0]
        auc = float(sum((p > n) + .5 * (p == n) for p in positive for n in negative)
                    / (len(positive) * len(negative)))
    return {"count": len(records), "good_count": good, "ng_count": ng,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "recheck_count": recheck,
            "good_recheck_count": sum(r["label"] == 0 and r["verdict"] == "RECHECK" for r in records),
            "ng_recheck_count": sum(r["label"] == 1 and r["verdict"] == "RECHECK" for r in records),
            "coverage": _ratio(decisive, len(records)), "recheck_rate": _ratio(recheck, len(records)),
            "decisive_accuracy": _ratio(tp + tn, decisive),
            "overall_correct_rate": _ratio(tp + tn, len(records)),
            "accuracy": _ratio(tp + tn, decisive),
            "precision": precision, "recall": recall,
            "specificity": _ratio(tn, tn + fp),
            "false_positive_rate": _ratio(fp, fp + tn),
            "false_negative_rate": _ratio(fn, fn + tp), "f1": f1,
            "good_part_false_rejection_rate": _ratio(fp, good),
            "defect_detection_rate_all_ng": _ratio(tp, ng),
            "false_positive_rate_95ci": wilson_interval(fp, fp + tn),
            "false_negative_rate_95ci": wilson_interval(fn, fn + tp),
            "recall_95ci": wilson_interval(tp, tp + fn),
            "coverage_95ci": wilson_interval(decisive, len(records)),
            "latency_median_ms": float(np.median(latencies)) if latencies else None,
            "latency_p95_ms": float(np.quantile(latencies, .95)) if latencies else None,
            "image_level_auroc": auc,
            "ng_sensitivity_validated": bool(ng),
            "metric_semantics": "Confusion metrics use decisive outcomes; coverage and GOOD/NG recheck counts include every sample. Rates for absent classes are null."}


def _aggregate_part(part_id: str, records: list[dict[str, Any]], mandatory: set[str]) -> dict[str, Any]:
    present = {r["view"] for r in records}
    missing = sorted(mandatory - present)
    # No retake is silently discarded: all captures remain in this blind evaluation.
    # Separate placements should carry different session IDs and appear in repeatability.
    verdicts = {r["verdict"] for r in records}
    verdict = "FAIL" if "FAIL" in verdicts else ("RECHECK" if missing or "RECHECK" in verdicts else "PASS")
    values = [r["score"] for r in records if r.get("score") is not None]
    return {"part_id": part_id, "label": max(r["label"] for r in records),
            "verdict": verdict, "missing_views": missing,
            "views": sorted(present), "captures": len(records),
            "score": max(values) if len(values) == len(records) else None,
            "latency_ms": sum(r["latency_ms"] for r in records)}


def evaluate_samples(samples: Iterable[Sample], inspect: Callable[[np.ndarray, str], Any], *,
                     mandatory_views: Iterable[str] = ("inner_face", "ribbed_face"),
                     score_key: str = "anomaly", training_part_ids: Iterable[str] = (),
                     validation_part_ids: Iterable[str] = (),
                     training_hashes: Iterable[str] = (), validation_hashes: Iterable[str] = (),
                     export_directory: str | Path | None = None) -> dict[str, Any]:
    """Invoke ``inspect(original_bgr, view)`` once per source image.

    Bind this callback to the same pipeline used live, including its quality,
    registration, localization and final decision logic. No threshold is fit here.
    Supply artifact provenance to enforce the held-out partition mechanically.
    Callback/image errors become RECHECK records, never omitted observations.
    """
    samples = list(samples)
    mandatory = set(mandatory_views)
    forbidden_parts = set(training_part_ids) | set(validation_part_ids)
    forbidden_hashes = set(training_hashes) | set(validation_hashes)
    if any(s.part_id in forbidden_parts or s.sha256 in forbidden_hashes for s in samples):
        raise DatasetIntegrityError("Blind evaluation overlaps model fitting/calibration/reference data")
    seen: dict[str, Sample] = {}
    for sample in samples:
        if sample.sha256 in seen:
            prior = seen[sample.sha256]
            if sample.part_id != prior.part_id or sample.label != prior.label or sample.view != prior.view:
                raise DatasetIntegrityError("Evaluation duplicates claim different physical parts, views or ground truth")
        seen[sample.sha256] = sample
    records: list[dict[str, Any]] = []
    for sample in samples:
        start = time.perf_counter()
        error = None
        score = None
        reasons: Any = []
        result = None
        try:
            import hashlib
            data = sample.path.read_bytes()
            if hashlib.sha256(data).hexdigest() != sample.sha256:
                raise DatasetIntegrityError("Original image hash changed after dataset validation")
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Unreadable original image")
            # This is the live callback; evaluation does not rerun an alternate classifier.
            result = inspect(frame, sample.view)
            verdict = _outcome(result)
            score = _score(result, score_key)
            reasons = _get(result, "reasons", _get(result, "reason", []))
        except Exception as exc:
            verdict, error = "RECHECK", f"{type(exc).__name__}: {exc}"
        elapsed = (time.perf_counter() - start) * 1000
        records.append({"sample_id": sample.id, "path": str(sample.path),
                        "sha256": sample.sha256, "view": sample.view,
                        "part_id": sample.part_id, "session_id": sample.session_id,
                        "label": sample.label, "verdict": verdict, "score": score,
                        "latency_ms": elapsed, "error": error, "reasons": reasons,
                        "defect_type": sample.metadata.get("defect_type")})
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    repeated: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    placements: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["part_id"]].append(record)
        repeated[(record["part_id"], record["view"])].append(record)
        if record["session_id"]:
            placements[(record["part_id"], record["session_id"])].append(record)
    parts = [_aggregate_part(key, values, mandatory) for key, values in sorted(groups.items())]
    repeatability = []
    for (part_id, view), values in sorted(repeated.items()):
        if len(values) < 2:
            continue
        counts = Counter(r["verdict"] for r in values)
        pairs = len(values) * (len(values)-1) // 2
        agreed = sum(c * (c-1) // 2 for c in counts.values())
        repeatability.append({"part_id": part_id, "view": view, "captures": len(values),
                              "verdict_counts": dict(counts), "pairwise_agreement": agreed/pairs,
                              "all_captures_consistent": len(counts) == 1,
                              "note": "Repeated placements are correlated evidence, not independent physical parts"})
    placement_records = [{**_aggregate_part(part, values, mandatory), "session_id": session}
                         for (part, session), values in sorted(placements.items())]
    placement_repeatability = []
    for part_id in sorted({r["part_id"] for r in placement_records}):
        values = [r for r in placement_records if r["part_id"] == part_id]
        if len(values) > 1:
            counts = Counter(r["verdict"] for r in values)
            pairs = len(values) * (len(values) - 1) // 2
            placement_repeatability.append({
                "part_id": part_id, "placements": len(values),
                "session_ids": [r["session_id"] for r in values],
                "verdict_counts": dict(counts),
                "pairwise_agreement": sum(c * (c - 1) // 2 for c in counts.values()) / pairs,
                "all_placements_consistent": len(counts) == 1,
            })
    per_defect: dict[str, Any] = {}
    for defect in sorted({r["defect_type"] for r in records if isinstance(r["defect_type"], str)}):
        tagged_parts = {r["part_id"] for r in records if r["defect_type"] == defect}
        relevant = [p for p in parts if p["part_id"] in tagged_parts and p["label"] == 1]
        detected = sum(p["verdict"] == "FAIL" for p in relevant)
        per_defect[defect] = {"independent_ng_parts": len(relevant),
                              "detected_parts": detected,
                              "sensitivity_all_parts": _ratio(detected, len(relevant)),
                              "sensitivity_95ci": wilson_interval(detected, len(relevant))}
    part_metrics = decision_metrics(parts)
    # Separate view-specific networks need not emit comparable raw-score scales.
    # Do not pool their scores into an apparently meaningful AUROC.
    image_metrics = decision_metrics(records)
    by_view = {view: decision_metrics(r for r in records if r["view"] == view)
               for view in sorted({r["view"] for r in records})}
    if len(by_view) > 1:
        image_metrics["image_level_auroc"] = None
        image_metrics["auroc_note"] = "View-specific raw-score scales are not pooled; see per_view AUROC"
        part_metrics["image_level_auroc"] = None
        part_metrics["auroc_note"] = "Multi-view part verdicts are evaluated by their confusion matrix; raw model scales are not pooled"
    report = {"schema_version": 1, "evaluation_mode": "production callback; no fitting or calibration",
              "mandatory_views": sorted(mandatory), "score_key": score_key,
              "provenance_checked": bool(forbidden_parts or forbidden_hashes),
              "independent_good_parts": part_metrics["good_count"],
              "independent_ng_parts": part_metrics["ng_count"],
              "parts": part_metrics, "images": image_metrics, "per_view": by_view,
              "records": records, "part_records": parts, "placement_records": placement_records,
              "repeatability": repeatability, "placement_repeatability": placement_repeatability,
              "per_defect_type": per_defect,
              "warnings": (["No NG test parts: false-negative rate and defect sensitivity are unknown"]
                           if not part_metrics["ng_count"] else [])
                          + (["Artifact fitting/calibration provenance was not supplied; held-out status is unverified"]
                             if not (forbidden_parts or forbidden_hashes) else []),
              "localization_metrics": None,
              "localization_note": "Pixel metrics require real ground-truth defect masks; none were supplied"}
    if export_directory is not None:
        export_report(report, export_directory)
    return report


def export_report(report: dict[str, Any], directory: str | Path) -> dict[str, str]:
    """Export all outcomes including RECHECK to JSON and image/part CSV files."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    result = {"json": str(folder / "evaluation.json"),
              "images_csv": str(folder / "images.csv"), "parts_csv": str(folder / "parts.csv")}
    target = Path(result["json"])
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)
    for key, records in (("images_csv", report.get("records", [])), ("parts_csv", report.get("part_records", []))):
        with Path(result[key]).open("w", newline="", encoding="utf-8-sig") as stream:
            columns = list(dict.fromkeys(column for record in records for column in record))
            if columns:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for record in records:
                    writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in record.items()})
    return result
