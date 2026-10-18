"""Service integration safety checks with explicit inspection/camera test seams.

These tests verify event/provenance behavior, not model accuracy or real hardware.
"""
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest
import yaml

from src.acquisition import FramePacket
from src.inspection.session import ViewDecision
from src.inspection.view_pipeline import Analysis, ViewPipeline
from src.service import NeuroIrisService


@pytest.fixture
def service(tmp_path):
    config = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text())
    config["paths"] = {name: tmp_path / name for name in config["paths"]}
    return NeuroIrisService(config)


def install_inspection_seam(service, monkeypatch, verdict="PASS"):
    def inspect(image, view, backend=None):
        return ViewDecision(view, verdict, ["Controlled service test seam"],
                            metrics={"inspection_scope": "geometry_only"},
                            model_version="test-artifact-id"), image.copy(), None
    monkeypatch.setattr(service, "_infer", inspect)


def test_one_part_event_has_both_originals_and_idempotent_count(service, monkeypatch):
    install_inspection_seam(service, monkeypatch)
    frame = np.full((31, 41, 3), 100, np.uint8)
    first = service.inspect_frame(frame, "inner_face", "part-native")
    assert first["session"]["verdict"] == "RECHECK"
    assert service.history() == []
    service.inspect_frame(frame + 1, "ribbed_face", "part-native")
    completed = service.finish_part()
    assert completed["verdict"] == "PASS"
    assert service.finish_part()["evidence_path"] == completed["evidence_path"]
    events = service.history()
    assert len(events) == 1 and events[0]["counted"] and events[0]["integrity_verified"]
    assert set(events[0]["images"]) == {"inner_face", "ribbed_face"}
    assert events[0]["images"]["inner_face"]["original"]["shape"] == [31, 41, 3]


def test_missing_model_stays_recheck_across_save_restart(service):
    result = service.inspect_frame(np.full((31, 41, 3), 100, np.uint8), "inner_face", "part-no-model")
    assert result["verdict"] == "RECHECK"
    final = service.finish_part()
    assert final["verdict"] == "RECHECK"
    restarted = NeuroIrisService(service.config)
    history = restarted.history()
    assert len(history) == 1 and history[0]["verdict"] == "RECHECK"
    assert "No trained" in history[0]["views"]["inner_face"]["reasons"][0]


def test_failed_evidence_commit_cannot_be_discarded_by_new_part_or_settings(service, monkeypatch):
    service.begin_part("part-unsaved")
    real_save = service.evidence.save
    def fail_save(*args, **kwargs):
        raise OSError("Controlled disk-full failure")
    monkeypatch.setattr(service.evidence, "save", fail_save)
    with pytest.raises(OSError, match="disk-full"):
        service.finish_part()
    with pytest.raises(RuntimeError, match="Evidence was not saved"):
        service.begin_part("part-next")
    with pytest.raises(RuntimeError, match="Evidence was not saved"):
        service.configure({"models": {"backend": "geometry"}})
    monkeypatch.setattr(service.evidence, "save", real_save)
    assert service.finish_part()["verdict"] == "RECHECK"
    service.begin_part("part-next")


def test_active_session_rejects_roi_model_and_runtime_changes(service):
    service.begin_part("part-config-lock")
    with pytest.raises(RuntimeError, match="current part"):
        service.save_rois("inner_face", [])
    with pytest.raises(RuntimeError, match="current part"):
        service.select_model("inner_face", "missing", "geometry")
    with pytest.raises(RuntimeError, match="current part"):
        service.configure({"models": {"backend": "geometry"}})


def test_wrong_part_identity_does_not_run_inference(service, monkeypatch):
    service.begin_part("part-A")
    def forbidden(*args, **kwargs):
        raise AssertionError("Inference must not run for the wrong physical part")
    monkeypatch.setattr(service, "_infer", forbidden)
    with pytest.raises(ValueError, match="active physical part"):
        service.inspect_frame(np.zeros((31, 41, 3), np.uint8), "inner_face", "part-B")


def test_auto_identity_recheck_does_not_poison_coverage_and_is_audited(service, monkeypatch):
    def unknown(image, view, backend=None):
        return ViewDecision("unidentified", "RECHECK", ["Face uncertain"]), image.copy(), None
    monkeypatch.setattr(service, "_infer", unknown)
    result = service.inspect_frame(np.zeros((31, 41, 3), np.uint8), "auto", "part-unknown")
    assert result["verdict"] == "RECHECK"
    assert service.session.decisions == {}
    service.cancel_part()
    assert service.history()[0]["attempts"][0]["decision"]["reasons"] == ["Face uncertain"]


def test_live_capture_provenance_matches_attempt_view_and_evidence(service, monkeypatch):
    install_inspection_seam(service, monkeypatch)
    service.begin_part("part-live")
    packet = FramePacket(np.full((31, 41, 3), 100, np.uint8), 3, time.monotonic())
    service.camera = SimpleNamespace(latest=lambda: packet, diagnostics=lambda: {"connected": True})
    monkeypatch.setattr("src.service.HousingLocator.detect", lambda *args: SimpleNamespace(mask=np.ones((31, 41), np.uint8)))
    monkeypatch.setattr("src.service.QualityChecker.select_sharpest", lambda *args, **kwargs: packet)
    result = service.inspect_live("inner_face", "part-live")
    assert result["metrics"]["capture_type"] == "live stable-frame selection"
    assert result["session"]["views"]["inner_face"]["metrics"]["capture_type"] == "live stable-frame selection"
    assert service.session.attempts[0]["decision"]["metrics"]["capture_type"] == "live stable-frame selection"
    service.cancel_part()
    assert service.history()[0]["attempts"][0]["decision"]["metrics"]["capture_type"] == "live stable-frame selection"


def test_invalid_live_capture_reason_is_audited_without_claiming_view(service, monkeypatch):
    service.begin_part("part-moving")
    packet = FramePacket(np.zeros((31, 41, 3), np.uint8), 3, time.monotonic())
    service.camera = SimpleNamespace(latest=lambda: packet, diagnostics=lambda: {"connected": True})
    monkeypatch.setattr("src.service.HousingLocator.detect", lambda *args: None)
    result = service.inspect_live("inner_face", "part-moving")
    assert result["verdict"] == "RECHECK"
    assert service.session.decisions == {}
    service.cancel_part()
    assert "stationary" in service.history()[0]["attempts"][0]["decision"]["reasons"][0]


def test_mapped_failure_cannot_survive_without_source_localization():
    pipeline = ViewPipeline.__new__(ViewPipeline)
    pipeline.view, pipeline.mode = "ribbed_face", "geometry"
    pipeline.model = None
    pipeline.metadata = {"version": "controlled-test"}
    pipeline.settings = {"registration": {"min_confidence": .55}}
    pipeline.thresholds = {"geometry": {"reject_threshold": 2., "recheck_threshold": 1.}}
    data = Analysis(valid=True,
                    quality=SimpleNamespace(metrics={}),
                    registration=SimpleNamespace(confidence=.9, inverse_matrix=np.eye(3)),
                    geometry=SimpleNamespace(valid=True, score=3., measurements={},
                                             candidates=[{"bbox": [1, 1, 5, 5], "score": 3.}]))
    pipeline.analyze = lambda image: data
    decision, marked, _ = pipeline.inspect(np.zeros((31, 41, 3), np.uint8))
    assert decision.verdict == "RECHECK"
    assert decision.candidates == []
