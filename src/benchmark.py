"""Complete-view evaluation and paired, fixed-artifact model comparisons.

Every mode uses the original held-out roster and the production inspection path.
No models, references, thresholds, or artifact pointers are changed here.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import json
import sys
import uuid
from typing import Any, Iterable

from src.artifacts import atomic_json
from src.data import DatasetIntegrityError, Sample
from src.evaluation import evaluate_samples, export_report
from src.utils.filesystem import display_path, filesystem_path


MODES = ("geometry", "efficientad", "hybrid")


def _views(service, view: str = "all") -> list[str]:
    views = list(service.config["views"]["required"]) if view == "all" else [view]
    if not views or len(views) != len(set(views)):
        raise ValueError("Evaluation requires unique, nonempty mandatory views")
    if not set(views) <= set(service.config["views"]["enabled"]):
        raise ValueError("An evaluation view is not enabled in the current product configuration")
    return views


def _roster(metadata: dict[str, Any]) -> dict[str, list[tuple[str, str, int, str, str]]]:
    try:
        split = metadata["split"]
        roster = {name: sorted((str(row["id"]), str(row["part_id"]), int(row["label"]),
                                str(row["view"]), str(row["sha256"]))
                               for row in split[name])
                  for name in ("train", "validation", "test")}
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetIntegrityError("A model bundle lacks a complete saved physical-part split manifest") from exc
    if any(not row[1].strip() for rows in roster.values() for row in rows):
        raise DatasetIntegrityError("A saved split contains an empty physical part ID")
    return roster


def _prepare(service, pipelines: dict[str, Any]) -> tuple[list[Sample], dict[str, Any]]:
    """Require an identical global capture roster and disjoint fit/test provenance."""
    if not pipelines:
        raise ValueError("No view pipelines were selected")
    reference = None
    reference_view = None
    training_parts: set[str] = set()
    validation_parts: set[str] = set()
    training_hashes: set[str] = set()
    validation_hashes: set[str] = set()
    test_parts: set[str] = set()
    test_hashes: set[str] = set()
    versions = {}
    artifacts = {}
    for view, pipeline in pipelines.items():
        meta = pipeline.metadata
        if meta.get("view") != view:
            raise DatasetIntegrityError(f"Active bundle identifies {meta.get('view')!r}, expected {view!r}")
        if meta.get("rois", []) != service.get_rois(view):
            raise ValueError(f"The {view} ROI configuration differs from its calibrated artifact; retrain before evaluation")
        roster = _roster(meta)
        if reference is None:
            reference, reference_view = roster, view
        elif roster != reference:
            raise DatasetIntegrityError(
                f"Saved split manifests disagree between {reference_view} and {view}. "
                "Train the views from the same dataset roster and physical-part partition; mixed rosters cannot form a paired blind evaluation.")
        split = meta["split"]
        training_parts.update(str(row["part_id"]) for row in split["train"])
        validation_parts.update(str(row["part_id"]) for row in split["validation"])
        # Declared provenance supplements (rather than replaces) actual manifest entries.
        training_parts.update(meta.get("training_parts", []))
        validation_parts.update(meta.get("validation_parts", []))
        training_hashes.update(row["sha256"] for row in split["train"])
        validation_hashes.update(row["sha256"] for row in split["validation"])
        test_parts.update(str(row["part_id"]) for row in split["test"])
        test_hashes.update(row["sha256"] for row in split["test"])
        versions[view] = meta.get("version")
        artifacts[view] = display_path(getattr(pipeline, "directory", ""))
    if test_parts & (training_parts | validation_parts) or test_hashes & (training_hashes | validation_hashes):
        raise DatasetIntegrityError("A held-out part or original image overlaps training/calibration in at least one active view model")
    # All rosters match, so one authoritative global manifest supplies every view.
    first = next(iter(pipelines.values()))
    rows = [row for row in first.metadata["split"]["test"] if row["view"] in pipelines]
    if not rows:
        raise ValueError("The saved untouched test partition contains no images for the requested views")
    if len({row["id"] for row in rows}) != len(rows):
        raise DatasetIntegrityError("The saved test roster contains repeated sample IDs")
    # This verifies original hashes before a potentially costly benchmark begins.
    samples = service._saved_samples(rows)
    signature = hashlib.sha256(json.dumps(reference, sort_keys=True).encode("utf-8")).hexdigest()
    return samples, {"training_part_ids": sorted(training_parts),
                     "validation_part_ids": sorted(validation_parts),
                     "training_hashes": sorted(training_hashes),
                     "validation_hashes": sorted(validation_hashes),
                     "model_versions": versions, "artifact_paths": artifacts,
                     "split_roster_sha256": signature}


def _arguments(provenance: dict[str, Any]) -> dict[str, Any]:
    return {key: provenance[key] for key in
            ("training_part_ids", "validation_part_ids", "training_hashes", "validation_hashes")}


def _scope(views: Iterable[str], all_views: bool) -> str:
    return ("Complete configured mandatory-view physical-part inspection: " if all_views else
            "Single-view held-out inspection; complete production sessions require all mandatory views: ") + ", ".join(views)


def evaluate_all(service, backend: str, progress=None) -> dict[str, Any]:
    """Evaluate every active mandatory-view model on one compatible blind roster."""
    if backend not in MODES:
        raise ValueError("Select geometry, efficientad, or hybrid for evaluation")
    emit = progress or (lambda *_: None)
    views = _views(service)
    emit(0, "Complete-view evaluation", "Checking active model provenance across every mandatory view")
    pipelines = {view: service._pipeline(view, backend) for view in views}
    samples, provenance = _prepare(service, pipelines)
    attempted = 0
    def inspect(image, view):
        nonlocal attempted
        try:
            # Same entry point as live still-image inspection, including its guards.
            return service._infer(image, view, backend)[0]
        finally:
            attempted += 1
            emit(min(95, round(95 * attempted / len(samples))), "Complete-view evaluation",
                 f"Processed {attempted}/{len(samples)} held-out captures")
    report = evaluate_samples(samples, inspect, mandatory_views=views,
                              score_key="geometry_score" if backend == "geometry" else "anomaly_score",
                              **_arguments(provenance))
    report.update({"scope": _scope(views, True), "backend": backend,
                   "model_versions": provenance["model_versions"],
                   "artifact_paths": provenance["artifact_paths"],
                   "split_roster_sha256": provenance["split_roster_sha256"]})
    output = filesystem_path(service.config["paths"]["results"]) / "evaluations" / uuid.uuid4().hex
    report["report_path"] = display_path(output)
    export_report(report, output)
    emit(100, "Evaluation completed", f"Actual complete-view outcomes saved to {output}")
    return report


def _memory_start() -> dict[str, Any]:
    """Reset CUDA high-water counters only when a real initialized CUDA exists."""
    torch = sys.modules.get("torch")
    result: dict[str, Any] = {"cuda_device": None, "cuda_peak_allocated_bytes": None,
                              "cuda_peak_reserved_bytes": None}
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        try:
            device = torch.cuda.current_device()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            result["cuda_device"] = device
        except (RuntimeError, AttributeError):
            pass
    return result


def _memory_finish(snapshot: dict[str, Any]) -> dict[str, Any]:
    torch = sys.modules.get("torch")
    if torch is not None and snapshot["cuda_device"] is not None:
        try:
            device = snapshot["cuda_device"]
            torch.cuda.synchronize(device)
            snapshot["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            snapshot["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        except (RuntimeError, AttributeError):
            pass
    snapshot["note"] = "Actual process CUDA high-water memory during this mode, including resident shared weights; null means no CUDA measurement was available"
    return snapshot


def benchmark(service, view: str, progress=None) -> dict[str, Any]:
    """Compare three real modes on identical samples using one EfficientAD bundle/view.

    Copies share read-only weights, registration, geometry and fixed calibration.
    Geometry mode releases its model reference and performs no neural inference.
    This prevents loading several copies of the same network into GPU memory.
    """
    emit = progress or (lambda *_: None)
    views = _views(service, view)
    emit(0, "Paired benchmark", "Loading one EfficientAD artifact per requested view; no model or threshold fitting")
    bases = {selected: service._pipeline(selected, "efficientad") for selected in views}
    for selected, pipeline in bases.items():
        if pipeline.metadata.get("backend") != "efficientad" or getattr(pipeline, "model", None) is None:
            raise ValueError(f"{selected} needs a genuine trained EfficientAD bundle for a paired three-mode benchmark")
    samples, provenance = _prepare(service, bases)
    output = filesystem_path(service.config["paths"]["results"]) / "benchmarks" / uuid.uuid4().hex
    reports = {}
    attempted = 0
    total = len(samples) * len(MODES)
    for mode in MODES:
        pipelines = {}
        for selected, base in bases.items():
            pipeline = copy.copy(base)
            pipeline.mode = mode
            if mode == "geometry":
                pipeline.model = None
            pipelines[selected] = pipeline
        memory = _memory_start()
        def inspect(image, selected):
            nonlocal attempted
            try:
                return service._infer_with_pipeline(image, selected, pipelines[selected])[0]
            finally:
                attempted += 1
                emit(min(95, round(95 * attempted / total)), "Paired benchmark",
                     f"{mode}: processed {attempted}/{total} total held-out inference calls")
        report = evaluate_samples(samples, inspect, mandatory_views=views,
                                  score_key="geometry_score" if mode == "geometry" else "anomaly_score",
                                  **_arguments(provenance))
        report.update({"scope": _scope(views, view == "all"), "backend": mode,
                       "model_versions": provenance["model_versions"],
                       "split_roster_sha256": provenance["split_roster_sha256"],
                       "fixed_shared_calibration": True, "memory": _memory_finish(memory),
                       "score_note": "Hybrid AUROC describes the anomaly branch score; fused decisions are assessed by confusion metrics, coverage and RECHECK rates" if mode == "hybrid" else "Selected branch raw score",
                       "report_path": display_path(output / mode)})
        export_report(report, output / mode)
        reports[mode] = report
    summary = {"schema_version": 1, "scope": _scope(views, view == "all"),
               "benchmark_mode": "paired fixed artifacts; no refitting or threshold selection",
               "views": views, "sample_ids": [s.id for s in samples],
               "model_versions": provenance["model_versions"],
               "artifact_paths": provenance["artifact_paths"],
               "split_roster_sha256": provenance["split_roster_sha256"],
               "modes": reports, "winner": None,
               "selection_note": "No automatic winner: compare held-out GOOD false rejects, defect sensitivity, RECHECK coverage and measured latency together. These modes share conservative fixed calibration.",
               "optional_models": {"dinomaly": "Not integrated or evaluated", "dinov2": "Disabled; not benchmarked"},
               "report_path": display_path(output),
               "warnings": ["Software benchmark output does not establish real housing accuracy or Windows camera acceptance"]}
    atomic_json(output / "benchmark.json", summary)
    emit(100, "Benchmark completed", f"Actual paired reports saved to {output}")
    return summary
