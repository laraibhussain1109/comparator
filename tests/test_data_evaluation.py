"""Software-only fixtures; these tests make no industrial accuracy claim."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data import DatasetIntegrityError, DatasetStore, Sample, split_by_part
from src.evaluation import (calibrate_operating_point, calibrate_threshold, decision_metrics,
                            evaluate_samples, localization_threshold)


def fixture_image(seed: int, shape=(60, 80, 3)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def add(store, seed, part_id, view="inner_face", label="GOOD", session_id=""):
    return store.capture_image(fixture_image(seed), view, label, part_id, session_id)


def test_native_capture_sidecar_and_safe_rejection(tmp_path):
    store = DatasetStore(tmp_path)
    image = fixture_image(1, (216, 384, 3))
    sample = store.capture_image(image, "inner_face", "GOOD", "physical-01", "day-one")
    original = sample.path.read_bytes()
    sidecar = sample.path.with_suffix(".png.json")
    metadata = sidecar.read_bytes()
    assert sample.sha256 == hashlib.sha256(original).hexdigest()
    assert sample.metadata["width"] == 384
    assert np.array_equal(cv2.imread(str(sample.path)), image)
    assert store.inventory()[0].session_id == "day-one"
    store.remove(sample.id)
    assert store.inventory() == []
    assert sample.path.read_bytes() == original
    assert sidecar.read_bytes() == metadata
    assert store.stats()["rejected"] == 1
    with pytest.raises(DatasetIntegrityError, match="duplicate"):
        store.capture_image(image, "inner_face", "NG", "different-part")


def test_import_preserves_source_bytes_and_needs_identity(tmp_path):
    image = tmp_path / "source.jpg"
    cv2.imwrite(str(image), fixture_image(7))
    store = DatasetStore(tmp_path / "dataset")
    sample = store.import_image(image, "ribbed_face", "NG", "part-21")
    assert sample.path.read_bytes() == image.read_bytes()
    with pytest.raises(ValueError, match="physical part"):
        store.import_image(image, "ribbed_face", "GOOD", "")
    with pytest.raises(ValueError, match="View"):
        store.import_image(image, "../escape", "GOOD", "x")


def test_unannotated_and_tampered_sources_fail_inventory(tmp_path):
    store = DatasetStore(tmp_path)
    sample = add(store, 1, "part")
    sample.path.write_bytes(cv2.imencode(".png", fixture_image(2))[1].tobytes())
    with pytest.raises(DatasetIntegrityError, match="immutable sidecar"):
        store.inventory()
    sample.path.unlink()
    sample.path.with_suffix(".png.json").unlink()
    cv2.imwrite(str(tmp_path / "inner_face" / "GOOD" / "unlabelled.png"), fixture_image(1))
    with pytest.raises(DatasetIntegrityError, match="Missing metadata/physical part ID"):
        store.inventory()


def test_scan_detects_same_hash_claiming_other_part_and_label(tmp_path):
    store = DatasetStore(tmp_path)
    first = add(store, 11, "actual-part")
    target = tmp_path / "inner_face" / "NG" / "copy.png"
    target.parent.mkdir()
    target.write_bytes(first.path.read_bytes())
    metadata = {**first.metadata, "id": "copy", "part_id": "invented-independent-part", "label": 1}
    target.with_suffix(".png.json").write_text(json.dumps(metadata))
    with pytest.raises(DatasetIntegrityError, match="Duplicate bytes"):
        store.inventory()


def test_similar_capture_warning_does_not_claim_independence(tmp_path):
    store = DatasetStore(tmp_path)
    image = fixture_image(6)
    store.capture_image(image, "inner_face", "GOOD", "one")
    image[0, 0] ^= 1
    store.capture_image(image, "inner_face", "GOOD", "two")
    assert store.stats()["near_duplicate_warnings"]


def test_split_groups_both_faces_and_repeated_captures_globally(tmp_path):
    store = DatasetStore(tmp_path)
    for part in range(12):
        for face, offset in (("inner_face", 0), ("ribbed_face", 100)):
            add(store, part + offset, f"part-{part}", face)
    split = split_by_part(store.inventory(), seed=4)
    memberships = [{s.part_id for s in getattr(split, name)} for name in ("train", "validation", "test")]
    assert all(memberships)
    assert not memberships[0] & memberships[1]
    assert not memberships[0] & memberships[2]
    assert not memberships[1] & memberships[2]
    assert len({s.sha256 for name in ("train", "validation", "test") for s in getattr(split, name)}) == 24
    assert split.manifest() == split_by_part(store.inventory(), seed=4).manifest()


def test_good_only_training_and_ng_part_ground_truth(tmp_path):
    store = DatasetStore(tmp_path)
    for part in range(6):
        add(store, part, f"good-{part}")
    add(store, 100, "bad-part", "inner_face", "GOOD")
    add(store, 101, "bad-part", "ribbed_face", "NG")
    split = split_by_part(store.inventory())
    assert not any(s.part_id == "bad-part" for s in split.train)
    assert any(s.part_id == "bad-part" for s in split.test)
    assert store.stats()["ng_parts"] == 1
    assert store.stats()["good_parts"] == 6


def test_split_needs_independent_groups_and_session_option(tmp_path):
    store = DatasetStore(tmp_path)
    for part in range(6):
        add(store, part, f"physical-{part}", session_id="same-acquisition")
    assert split_by_part(store.inventory()).train
    with pytest.raises(ValueError, match="independent GOOD"):
        split_by_part(store.inventory(), group_sessions=True)
    with pytest.raises(ValueError, match="ratios"):
        split_by_part(store.inventory(), ratios=(1., 0., 0.))


def test_calibration_is_false_positive_constrained_and_honest_without_ng():
    scores = np.arange(100, dtype=float)
    point = calibrate_operating_point(scores, good_part_ids=[f"part-{i}" for i in range(100)],
                                      target_fpr=.02, recheck_fpr=.1)
    assert sum(point.decide(x) == "FAIL" for x in scores) == 2
    assert sum(point.decide(x) != "PASS" for x in scores) == 10
    assert point.report["false_negative_rate"] is None
    assert point.report["defect_recall"] is None
    assert not point.report["ng_sensitivity_validated"]
    assert point.report["false_positive_rate_95ci"][1] > .02
    assert point.decide(float("nan")) == "RECHECK"


def test_calibration_counts_physical_parts_and_refuses_leakage():
    point = calibrate_operating_point([1., 1.1, 2., 3.], good_part_ids=["a", "a", "b", "c"],
                                      target_fpr=0., recheck_fpr=0.)
    assert point.report["independent_good_parts"] == 3
    assert point.reject_threshold > 3.
    with pytest.raises(ValueError, match="IDs"):
        calibrate_operating_point([1., 2., 3.])
    with pytest.raises(ValueError, match="overlap"):
        calibrate_operating_point([1.], good_part_ids=["a"], training_part_ids=["a"])
    threshold, report = calibrate_threshold([0, 0, 0, 1], [1., 2., 3., 4.],
                                          part_ids=["a", "b", "c", "d"], target_fpr=0)
    assert 3. < threshold < 4.
    assert report["defect_recall"] == 1.


def test_localization_threshold_preserves_between_image_magnitude():
    maps = [np.ones((10, 10)), np.full((10, 10), 10.)]
    assert localization_threshold(maps, .9) == 10.
    assert localization_threshold([m * 3 for m in maps], .9) == 30.
    with pytest.raises(ValueError):
        localization_threshold([])


def test_evaluation_runs_exact_production_callback_and_includes_rechecks(tmp_path):
    store = DatasetStore(tmp_path / "data")
    samples = [add(store, 1, "good", "inner_face"), add(store, 2, "good", "ribbed_face"),
               add(store, 3, "bad", "inner_face", "NG"), add(store, 4, "bad", "ribbed_face", "GOOD"),
               add(store, 5, "incomplete", "inner_face")]
    seen = []
    def production(frame, view):
        index = len(seen)
        seen.append((frame.copy(), view))
        return {"verdict": "FAIL" if index == 2 else "PASS", "scores": {"anomaly": float(index)}}
    report = evaluate_samples(samples, production, training_part_ids=["other-training-part"],
                              export_directory=tmp_path / "report")
    assert len(seen) == len(samples)
    assert all(np.array_equal(actual[0], cv2.imread(str(sample.path))) and actual[1] == sample.view
               for actual, sample in zip(seen, samples))
    assert report["parts"]["TP"] == 1
    assert report["parts"]["TN"] == 1
    assert report["parts"]["recheck_count"] == 1
    assert report["parts"]["coverage"] == pytest.approx(2 / 3)
    assert report["parts"]["decisive_accuracy"] == 1.
    assert report["parts"]["overall_correct_rate"] == pytest.approx(2 / 3)
    assert len(report["part_records"]) == 3
    assert "RECHECK" in (tmp_path / "report" / "parts.csv").read_text(encoding="utf-8-sig")
    assert json.loads((tmp_path / "report" / "evaluation.json").read_text())["parts"]["TP"] == 1


def test_evaluation_errors_and_good_only_do_not_fabricate_sensitivity(tmp_path):
    store = DatasetStore(tmp_path)
    samples = [add(store, 1, "part")]
    def broken(frame, view):
        raise RuntimeError("teacher checkpoint unavailable")
    report = evaluate_samples(samples, broken)
    assert report["images"]["recheck_count"] == 1
    assert "checkpoint unavailable" in report["records"][0]["error"]
    assert report["parts"]["false_negative_rate"] is None
    assert report["parts"]["recall"] is None
    assert report["parts"]["image_level_auroc"] is None
    assert report["parts"]["decisive_accuracy"] is None


def test_evaluation_refuses_fit_data_and_detects_repeatability(tmp_path):
    store = DatasetStore(tmp_path)
    samples = [add(store, 1, "part"), add(store, 2, "part")]
    with pytest.raises(DatasetIntegrityError, match="overlaps"):
        evaluate_samples(samples, lambda f, v: {"verdict": "PASS"}, training_part_ids=["part"])
    verdicts = iter(["PASS", "RECHECK"])
    report = evaluate_samples(samples, lambda f, v: {"verdict": next(verdicts)})
    assert report["repeatability"][0]["pairwise_agreement"] == 0
    assert report["parts"]["recheck_count"] == 1
    assert report["independent_good_parts"] == 1


def test_evaluation_reads_production_view_decision_metrics(tmp_path):
    from src.inspection.session import ViewDecision
    store = DatasetStore(tmp_path)
    samples = [add(store, 1, "one")]
    report = evaluate_samples(
        samples, lambda f, v: ViewDecision(v, "PASS", metrics={"geometry_score": 2.5}),
        mandatory_views=["inner_face"], score_key="geometry_score")
    assert report["records"][0]["score"] == 2.5
    assert report["parts"]["TN"] == 1


def test_metrics_auroc_undefined_and_recheck_denominators():
    metrics = decision_metrics([{"label": 0, "verdict": "PASS", "score": .2},
                                {"label": 1, "verdict": "FAIL", "score": .8},
                                {"label": 1, "verdict": "RECHECK", "score": .5}])
    assert metrics["image_level_auroc"] == 1.
    assert metrics["recall"] == 1.
    assert metrics["defect_detection_rate_all_ng"] == .5
    assert metrics["ng_recheck_count"] == 1
    assert metrics["coverage"] == pytest.approx(2 / 3)


def test_geometry_measurement_floor_recalculates_reported_ng_tradeoff():
    from src.evaluation import calibrate_operating_point
    op = calibrate_operating_point([0.0, 0.0], good_part_ids=['g1', 'g2'],
                                   ng_scores=[.5, 2.0], ng_part_ids=['n1', 'n2'],
                                   minimum_reject_score=1.0, minimum_recheck_score=.8)
    assert op.reject_threshold == 1.0 and op.recheck_threshold == .8
    assert op.report['defect_recall'] == .5
    assert op.report['false_negative_rate'] == .5
    assert op.decide(.5) == 'PASS' and op.decide(2.0) == 'FAIL'
