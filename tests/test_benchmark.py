"""Paired-evaluation orchestration against controlled software fixtures only."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.benchmark import benchmark, evaluate_all
from src.data import DatasetIntegrityError, DatasetStore, Sample, split_by_part
from src.inspection.session import ViewDecision


class FixturePipeline:
    def __init__(self, owner, view, metadata):
        self.owner = owner
        self.view = view
        self.metadata = metadata
        self.directory = Path("controlled-fixture") / view
        self.mode = "efficientad"
        # This marker is injected by a fake service; no network is presented as trained.
        self.model = object()

    def inspect(self, frame):
        self.owner.pipeline_calls.append((self.view, self.mode, self.model))
        defect_marker = int(frame[0, 0, 0]) == 200
        verdict = "FAIL" if defect_marker and self.mode != "geometry" else "PASS"
        return ViewDecision(self.view, verdict, metrics={
            "geometry_score": .1, "anomaly_score": .9 if defect_marker else .1}), frame, None


class FixtureService:
    def __init__(self, root):
        self.config = {"views": {"required": ["inner_face", "ribbed_face"],
                                 "enabled": ["inner_face", "ribbed_face"]},
                       "paths": {"results": str(root / "results")}}
        store = DatasetStore(root / "dataset")
        seed = 0
        for part in range(4):
            for view in self.config["views"]["required"]:
                seed += 1
                image = np.random.default_rng(seed).integers(0, 256, (30, 40, 3), dtype=np.uint8)
                image[0, 0, 0] = 200 if part == 3 else 10
                store.capture_image(image, view, "NG" if part == 3 else "GOOD", f"physical-{part}")
        self.manifest = split_by_part(store.inventory()).manifest()
        self.pipeline_calls = []
        self.live_calls = []
        self.shared_calls = []
        self.rois = {view: [] for view in self.config["views"]["required"]}
        self.pipelines = {}
        for view in self.config["views"]["required"]:
            metadata = {"backend": "efficientad", "version": f"fixture-{view}", "view": view,
                        "rois": [], "split": copy.deepcopy(self.manifest),
                        "training_parts": sorted({r["part_id"] for r in self.manifest["train"]}),
                        "validation_parts": sorted({r["part_id"] for r in self.manifest["validation"]})}
            self.pipelines[view] = FixturePipeline(self, view, metadata)

    def _pipeline(self, view, backend):
        return self.pipelines[view]

    def get_rois(self, view):
        return self.rois[view]

    def _saved_samples(self, rows):
        samples = []
        for row in rows:
            path = Path(row["path"])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Original held-out sample missing or modified")
            samples.append(Sample(row["id"], path, row["label"], row["view"], row["part_id"],
                                  row["sha256"], row["metadata"]))
        return samples

    def _infer(self, frame, view, backend):
        self.live_calls.append((view, backend))
        return self._infer_with_pipeline(frame, view, self.pipelines[view])

    def _infer_with_pipeline(self, frame, view, pipeline):
        assert pipeline.view == view
        self.shared_calls.append((view, pipeline.mode, pipeline))
        return pipeline.inspect(frame)


def test_complete_view_evaluation_uses_live_callback_and_exports(tmp_path):
    service = FixtureService(tmp_path)
    progress = []
    report = evaluate_all(service, "hybrid", lambda *args: progress.append(args))
    assert len(service.live_calls) == len(service.manifest["test"])
    assert all(backend == "hybrid" for _, backend in service.live_calls)
    assert report["mandatory_views"] == ["inner_face", "ribbed_face"]
    assert report["parts"]["TP"] == 1
    assert report["parts"]["TN"] == 1
    assert report["parts"]["coverage"] == 1
    assert report["provenance_checked"]
    output = Path(report["report_path"])
    assert (output / "evaluation.json").is_file()
    assert (output / "parts.csv").is_file()
    assert len(report["model_versions"]) == 2
    assert progress[-1][0] == 100


def test_complete_view_missing_capture_is_recheck(tmp_path):
    service = FixtureService(tmp_path)
    good_part = next(r["part_id"] for r in service.manifest["test"] if r["label"] == 0)
    for pipeline in service.pipelines.values():
        pipeline.metadata["split"]["test"] = [r for r in pipeline.metadata["split"]["test"]
                                               if not (r["part_id"] == good_part and r["view"] == "ribbed_face")]
    report = evaluate_all(service, "hybrid")
    assert report["parts"]["recheck_count"] == 1
    assert report["parts"]["TP"] == 1
    assert report["parts"]["coverage"] == .5


def test_complete_view_rejects_mismatched_rosters_and_cross_model_leakage(tmp_path):
    service = FixtureService(tmp_path)
    service.pipelines["ribbed_face"].metadata["split"]["test"].pop()
    with pytest.raises(DatasetIntegrityError, match="split manifests disagree"):
        evaluate_all(service, "hybrid")
    assert not service.pipeline_calls
    service = FixtureService(tmp_path / "second")
    heldout = service.manifest["test"][0]["part_id"]
    service.pipelines["ribbed_face"].metadata["training_parts"].append(heldout)
    with pytest.raises(DatasetIntegrityError, match="overlaps training/calibration"):
        evaluate_all(service, "hybrid")
    assert not service.pipeline_calls


def test_benchmark_reuses_one_bundle_and_identical_test_samples(tmp_path):
    service = FixtureService(tmp_path)
    before = {view: (pipeline.mode, pipeline.model, copy.deepcopy(pipeline.metadata))
              for view, pipeline in service.pipelines.items()}
    result = benchmark(service, "all")
    assert set(result["modes"]) == {"geometry", "efficientad", "hybrid"}
    assert result["winner"] is None
    assert "Not integrated" in result["optional_models"]["dinomaly"]
    assert len(service.shared_calls) == len(service.manifest["test"]) * 3
    assert not service.live_calls
    rosters = [{r["sample_id"] for r in report["records"]} for report in result["modes"].values()]
    assert rosters[0] == rosters[1] == rosters[2] == set(result["sample_ids"])
    assert result["modes"]["geometry"]["parts"]["FN"] == 1
    assert result["modes"]["efficientad"]["parts"]["TP"] == 1
    for view, mode, model in service.pipeline_calls:
        if mode == "geometry":
            assert model is None
        else:
            assert model is before[view][1]
    for view, pipeline in service.pipelines.items():
        assert (pipeline.mode, pipeline.model, pipeline.metadata) == before[view]
    summary = json.loads((Path(result["report_path"]) / "benchmark.json").read_text())
    assert summary["modes"]["hybrid"]["parts"]["TP"] == 1
    assert summary["modes"]["geometry"]["memory"]["cuda_peak_allocated_bytes"] is None


def test_benchmark_single_view_scope_and_current_rois(tmp_path):
    service = FixtureService(tmp_path)
    result = benchmark(service, "inner_face")
    assert result["views"] == ["inner_face"]
    assert all(r["view"] == "inner_face" for r in result["modes"]["geometry"]["records"])
    assert "Single-view" in result["scope"]
    service.rois["inner_face"] = [{"shape": "rect"}]
    with pytest.raises(ValueError, match="ROI configuration differs"):
        benchmark(service, "inner_face")


def test_benchmark_refuses_missing_neural_model_and_tampered_original(tmp_path):
    service = FixtureService(tmp_path)
    service.pipelines["inner_face"].model = None
    with pytest.raises(ValueError, match="genuine trained EfficientAD"):
        benchmark(service, "all")
    service = FixtureService(tmp_path / "second")
    Path(service.manifest["test"][0]["path"]).write_bytes(b"changed original")
    with pytest.raises(ValueError, match="missing or modified"):
        benchmark(service, "all")
    assert not service.pipeline_calls
