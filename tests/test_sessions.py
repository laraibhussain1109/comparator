"""Software behavior fixtures only; these are not housing-accuracy measurements."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.inspection.evidence import EvidenceStore
from src.inspection.fusion import EvidenceDecisionEngine
from src.inspection.metrology import (ConstantExternalLighting, DisabledMetrologyProvider,
                                     Measurement, ScaleCalibration)
from src.inspection.session import InspectionSession, SessionState, ViewDecision


def accepted(view="inner_face", scope="hybrid"):
    return ViewDecision(view, "PASS", ["Calibrated checks passed"], metrics={"inspection_scope": scope},
                        model_version="tested-model-v1")


def structural_candidate():
    return {"kind": "missing rib", "bbox": [10, 12, 8, 9],
            "contour": [[10, 12], [18, 12], [18, 21], [10, 21]],
            "score": 3.2, "source": "GEOMETRY"}


def geometry(score=0.2, candidates=None, valid=True):
    return SimpleNamespace(valid=valid, score=score, candidates=candidates or [],
                           measurements={"deviation_px": score * 2})


def calibrated_anomaly(score=0.1, candidates=None):
    return {"available": True, "calibrated": True, "score": score,
            "threshold": 0.8, "recheck_threshold": 0.5, "localization_threshold": 0.6,
            "candidates": candidates or [], "model_version": "efficientad-v4"}


def fuse(geom=None, anomaly=None, **kwargs):
    cfg = {"registration_min_confidence": .6, "geometry_reject_threshold": 2.,
           "geometry_recheck_threshold": 1.}
    cfg.update(kwargs.pop("settings", {}))
    return EvidenceDecisionEngine().decide("ribbed_face", kwargs.pop("quality_valid", True),
                                           kwargs.pop("registration_confidence", .9),
                                           geom if geom is not None else geometry(),
                                           anomaly if anomaly is not None else calibrated_anomaly(),
                                           settings=cfg, **kwargs)


def test_mandatory_views_and_counting_are_part_level():
    session = InspectionSession("physical-17")
    assert session.preview().verdict == "RECHECK"
    assert session.record(accepted()).verdict == "RECHECK"
    assert session.missing_views == ["ribbed_face"]
    assert session.state == SessionState.WAITING_FOR_FLIP
    assert not session.counted
    assert session.record(accepted("ribbed_face")).verdict == "PASS"
    assert session.state == SessionState.READY_TO_FINALIZE
    first = session.finalize()
    assert first.verdict == "PASS" and session.counted
    assert session.finalize() == first
    assert len([a for a in session.attempts if a["action"] == "finalize"]) == 1
    with pytest.raises(RuntimeError):
        session.record(accepted("side_left"))


def test_finalizing_missing_view_never_passes():
    session = InspectionSession("part-18")
    session.record(accepted())
    assert session.finalize().verdict == "RECHECK"
    assert session.counted


def test_only_invalid_capture_can_be_retaken_and_audit_is_preserved():
    session = InspectionSession("part-19")
    session.record(ViewDecision("inner_face", "RECHECK", ["Motion detected"]))
    session.record(accepted())
    assert session.attempts[0]["decision"]["reasons"] == ["Motion detected"]
    assert session.attempts[1]["action"] == "retake"
    with pytest.raises(ValueError, match="invalid RECHECK"):
        session.record(accepted())


def test_failure_locks_until_explicit_reset_or_cancel():
    session = InspectionSession("part-20")
    failure = ViewDecision("ribbed_face", "FAIL", ["Broken rib"], [structural_candidate()])
    session.record(failure)
    failure.candidates.clear()
    assert session.preview().candidates
    assert session.state == SessionState.FAIL_LOCKED
    with pytest.raises(RuntimeError):
        session.record(accepted())
    assert session.cancel("Operator removed part").verdict == "RECHECK"
    assert not session.counted
    assert session.finalize().metrics["prior_verdict"] == "FAIL"
    old_id = session.event_id
    session.reset("Correct fixture and restart")
    assert session.event_id != old_id
    assert session.state == SessionState.AWAITING_VIEWS
    assert session.reset_history[-1]["event"]["cancel_reason"] == "Operator removed part"


@pytest.mark.parametrize("kwargs", [{"quality_valid": False}, {"registration_confidence": .2},
                                    {"registration_confidence": float("nan")},
                                    {"registration_confidence": 1.2}])
def test_bad_capture_or_registration_cannot_fail_or_pass(kwargs):
    assert fuse(geometry(5, [structural_candidate()]), **kwargs).verdict == "RECHECK"


def test_geometry_rejection_is_calibrated_and_localized():
    decision = fuse(geometry(3, [structural_candidate()]))
    assert decision.verdict == "FAIL"
    assert decision.candidates[0]["kind"] == "missing rib"
    assert fuse(geometry(3)).verdict == "RECHECK"
    assert fuse(geometry(3, [structural_candidate()]), settings={"geometry_calibrated": False}).verdict == "RECHECK"


def test_surface_rejection_requires_calibrated_spatial_evidence():
    high = calibrated_anomaly(.95)
    assert fuse(anomaly=high).verdict == "RECHECK"
    high["candidates"] = [{**structural_candidate(), "source": "efficientad", "peak_score": .9}]
    decision = fuse(anomaly=high)
    assert decision.verdict == "FAIL" and decision.model_version == "efficientad-v4"
    high["calibrated"] = False
    assert fuse(anomaly=high).verdict == "RECHECK"


def test_maps_use_absolute_thresholds_and_real_coordinates():
    high = calibrated_anomaly(.95)
    values = np.zeros((80, 100), np.float32)
    values[30:40, 50:60] = .9
    high["map"] = values
    result = fuse(anomaly=high)
    assert result.verdict == "FAIL"
    assert result.candidates[0]["bounding_box"] == [50, 30, 10, 10]
    assert result.candidates[0]["peak_score"] == pytest.approx(.9)


def test_missing_model_and_ambiguous_or_subthreshold_regions_require_recheck():
    assert fuse(anomaly={"available": False}).verdict == "RECHECK"
    assert fuse(anomaly=calibrated_anomaly(.6)).verdict == "RECHECK"
    unresolved = calibrated_anomaly(.2, [{**structural_candidate(), "peak_score": .9}])
    assert fuse(anomaly=unresolved).verdict == "RECHECK"
    assert fuse(geometry(1.3)).verdict == "RECHECK"
    assert fuse(geometry(valid=False)).verdict == "RECHECK"
    assert fuse().verdict == "PASS"


def test_invalid_raw_scores_or_missing_model_availability_never_pass():
    assert fuse(geometry(-1)).verdict == "RECHECK"
    assert fuse(anomaly=calibrated_anomaly(-1)).verdict == "RECHECK"
    missing_status = calibrated_anomaly()
    missing_status.pop("available")
    assert fuse(anomaly=missing_status).verdict == "RECHECK"


def test_geometry_only_pass_discloses_no_surface_coverage():
    view = fuse(mode="geometry_only", anomaly={"available": False})
    assert view.verdict == "PASS" and "surface defects were not assessed" in view.reasons[0]
    session = InspectionSession("geometry-test")
    session.record(accepted(scope="geometry_only"))
    session.record(accepted("ribbed_face", scope="geometry_only"))
    assert "surface defects were not assessed" in session.finalize().reasons[0]


def saved_session():
    session = InspectionSession("physical-21", ("inner_face",))
    decision = accepted()
    decision.candidates = [structural_candidate()]
    session.record(decision)
    session.finalize()
    return session


def test_native_lossless_evidence_roundtrip_and_idempotence(tmp_path):
    session = saved_session()
    frame = np.arange(73 * 101 * 3, dtype=np.uint16).reshape(73, 101, 3)
    marked = frame.copy()
    marked[1:3] = 65535
    store = EvidenceStore(tmp_path)
    folder = store.save(session, {"inner_face": frame}, {"inner_face": marked})
    assert store.save(session, {"inner_face": frame}, {"inner_face": marked}) == folder
    recovered = cv2.imread(str(folder / "inner_face_original.png"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(recovered, frame)
    record = store.history()[0]
    assert record["integrity_verified"]
    assert record["views"]["inner_face"]["model_version"] == "tested-model-v1"
    assert record["views"]["inner_face"]["candidates"][0]["contour"] == structural_candidate()["contour"]
    changed = frame.copy()
    changed[0, 0] = 1
    with pytest.raises(FileExistsError, match="not overwritten"):
        store.save(session, {"inner_face": changed}, {"inner_face": marked})
    assert len(store.history()) == 1


def test_cancellation_and_recheck_reasons_are_real_history(tmp_path):
    session = InspectionSession("physical-22")
    session.record(ViewDecision("inner_face", "RECHECK", ["Too blurred"]))
    session.cancel("Camera disconnected during flip")
    store = EvidenceStore(tmp_path)
    store.save(session, {"inner_face": np.zeros((7, 9, 3), np.uint8)})
    record = store.history()[0]
    assert record["state"] == "CANCELLED"
    assert not record["counted"]
    assert record["final_decision"]["reasons"] == ["Camera disconnected during flip"]
    assert record["attempts"][0]["decision"]["reasons"] == ["Too blurred"]


def test_unassigned_invalid_attempts_are_audited_without_claiming_coverage(tmp_path):
    session = InspectionSession("physical-audit")
    session.record_attempt(ViewDecision("unidentified", "RECHECK", ["Face identity uncertain"]))
    session.record_attempt(ViewDecision("inner_face", "RECHECK", ["Moving housing"]), "unstable_live")
    assert session.missing_views == ["inner_face", "ribbed_face"]
    assert not session.decisions
    assert session.invalid_views == []
    session.cancel("Part removed")
    store = EvidenceStore(tmp_path)
    store.save(session, {})
    reasons = [a["decision"]["reasons"][0] for a in store.history()[0]["attempts"] if "decision" in a]
    assert reasons == ["Face identity uncertain", "Moving housing"]
    with pytest.raises(RuntimeError, match="locked"):
        session.record_attempt(ViewDecision("inner_face", "RECHECK", ["Too late"]))


def test_coverage_neutral_attempt_cannot_hide_pass_or_failure():
    session = InspectionSession("physical-audit-invalid")
    with pytest.raises(ValueError, match="RECHECK"):
        session.record_attempt(ViewDecision("inner_face", "FAIL", ["Broken housing"]))


def test_failed_evidence_write_is_not_published(tmp_path, monkeypatch):
    store = EvidenceStore(tmp_path)
    session = saved_session()
    monkeypatch.setattr(cv2, "imencode", lambda *args: (False, None))
    with pytest.raises(OSError):
        store.save(session, {"inner_face": np.zeros((7, 9, 3), np.uint8)})
    assert store.history() == []
    assert list(tmp_path.iterdir()) == []


def test_evidence_corruption_is_exposed_and_never_overwritten(tmp_path):
    store = EvidenceStore(tmp_path)
    session = saved_session()
    frames = {"inner_face": np.zeros((7, 9, 3), np.uint8)}
    folder = store.save(session, frames)
    (folder / "inner_face_original.png").write_bytes(b"damaged evidence")
    assert store.history()[0]["integrity_verified"] is False
    with pytest.raises(ValueError, match="integrity"):
        store.save(session, frames)


def test_copying_event_to_new_uuid_does_not_count_duplicate_as_verified(tmp_path):
    from shutil import copytree
    from uuid import uuid4
    store = EvidenceStore(tmp_path)
    session = saved_session()
    folder = store.save(session, {"inner_face": np.zeros((7, 9, 3), np.uint8)})
    copytree(folder, tmp_path / str(uuid4()))
    assert sum(record["integrity_verified"] for record in store.history()) == 1


def test_unsupported_evidence_schema_has_explicit_corruption_record(tmp_path):
    import hashlib
    import json
    store = EvidenceStore(tmp_path)
    session = saved_session()
    folder = store.save(session, {"inner_face": np.zeros((7, 9, 3), np.uint8)})
    event = json.loads((folder / "event.json").read_text())
    event["schema_version"] = 999
    raw = json.dumps(event).encode()
    (folder / "event.json").write_bytes(raw)
    (folder / "event.sha256").write_text(hashlib.sha256(raw).hexdigest())
    record = store.history()[0]
    assert record["integrity_verified"] is False
    assert "schema" in record["storage_error"]


def test_missing_original_and_resized_marked_frame_are_rejected(tmp_path):
    store = EvidenceStore(tmp_path)
    session = saved_session()
    with pytest.raises(ValueError, match="Original evidence missing"):
        store.save(session, {})
    with pytest.raises(ValueError, match="original frame dimensions"):
        store.save(session, {"inner_face": np.zeros((9, 9, 3), np.uint8)},
                   {"inner_face": np.zeros((4, 4, 3), np.uint8)})


def test_physical_units_require_real_matching_validated_calibration():
    raw = Measurement("rim deviation", 4., uncertainty=.5)
    calibration = ScaleCalibration("actual-scale-1", "top-camera", (3840, 2160), 20.)
    with pytest.raises(ValueError, match="validated calibration"):
        calibration.to_mm(raw, camera_id="top-camera", image_size=(3840, 2160))
    calibration = ScaleCalibration("actual-scale-1", "top-camera", (3840, 2160), 20.,
                                   validated=True, scale_relative_uncertainty=.02)
    result = calibration.to_mm(raw, camera_id="top-camera", image_size=(3840, 2160))
    assert result.value == .2 and result.unit == "mm"
    assert result.uncertainty > .025 and result.calibration_id == "actual-scale-1"
    with pytest.raises(ValueError, match="native image size"):
        calibration.to_mm(raw, camera_id="top-camera", image_size=(1920, 1080))


def test_unavailable_hardware_never_returns_fake_readings():
    assert not DisabledMetrologyProvider.available
    with pytest.raises(RuntimeError, match="No real metrology"):
        DisabledMetrologyProvider().measure("actual-part")
    lighting = ConstantExternalLighting()
    assert not lighting.programmable
    lighting.select_condition("constant_external")
    with pytest.raises(RuntimeError, match="hardware"):
        lighting.select_condition("photometric_stereo")
