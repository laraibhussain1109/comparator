"""Training-only capture screening with real geometry and software housings.

These are executable integrity/workflow checks, not industrial accuracy results.
Invalid captures remain immutable dataset records and held-out observations.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.acquisition import QualityChecker
from src.artifacts import ArtifactStore
from src.data import DatasetIntegrityError, split_by_part
from src.service import NeuroIrisService
from src.utils.image_utils import read_image
from src.vision import GeometryProfile, HousingLocator, LandmarkRegistrar, Registration
from test_smoke_training import import_housings, make_service
from test_vision import housing


VIEW = "ribbed_face"


def dataset_bytes(service):
    root = service.dataset.root
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def current_split(service):
    return split_by_part(service.dataset.inventory(), seed=service.config["runtime"]["random_seed"],
                         ratios=tuple(service.config.get("dataset", {}).get("split", [.7, .15, .15])),
                         group_sessions=service.config.get("dataset", {}).get("group_sessions", False))


def good_training(split):
    return [sample for sample in split.train if sample.view == VIEW and sample.label == 0]


def assert_original_roster_and_actual_training(metadata, split, accepted_ids):
    for partition in ("train", "validation", "test"):
        assert {row["id"] for row in metadata["split"][partition]} == {
            sample.id for sample in getattr(split, partition)}
    training_ids = {sample.id for sample in good_training(split)}
    held_out_ids = {sample.id for sample in split.validation + split.test}
    ng_ids = {sample.id for sample in split.train + split.validation + split.test if sample.label == 1}
    assert set(metadata["training_accepted_sample_ids"]) == set(accepted_ids)
    assert set(accepted_ids) <= training_ids
    assert not set(accepted_ids) & (held_out_ids | ng_ids)


def assert_screening_persisted(service, report):
    persisted = service.training_screening(VIEW)
    assert persisted["status"]
    assert persisted["accepted_sample_ids"] == report["accepted_sample_ids"]
    assert persisted["excluded"] == report["excluded"]
    path = Path(persisted["report_path"])
    assert path.is_file()
    disk = json.loads(path.read_text(encoding="utf-8"))
    assert disk["accepted_sample_ids"] == report["accepted_sample_ids"]
    return persisted


def blurred_capture(service, part_id, tag):
    image = cv2.GaussianBlur(housing(True), (25, 25), 5)
    image[0, tag] = [tag + 1, 4, 9]  # Distinct file bytes outside the inspected part.
    pose = HousingLocator().detect(image)
    report = QualityChecker(service.config["quality"]).check(image, pose.mask if pose else None)
    assert pose is not None and "insufficient_focus" in report.reasons
    return service.capture_sample(image, VIEW, "GOOD", part_id, "software-fixture")


def test_blurred_train_capture_is_excluded_while_held_out_captures_remain_observations(tmp_path):
    service = make_service(tmp_path)
    import_housings(service)
    initial = current_split(service)
    train_part = good_training(initial)[0].part_id
    validation_part = next(sample.part_id for sample in initial.validation if sample.view == VIEW and sample.label == 0)
    test_part = next(sample.part_id for sample in initial.test if sample.view == VIEW and sample.label == 0)
    bad_train = blurred_capture(service, train_part, 11)
    bad_validation = blurred_capture(service, validation_part, 12)
    bad_test = blurred_capture(service, test_part, 13)
    service.capture_sample(housing(True, missing=10), VIEW, "NG", "controlled-ng-fixture", "software-fixture")
    split = current_split(service)
    candidates = good_training(split)
    assert bad_train["id"] in {sample.id for sample in candidates}
    immutable = dataset_bytes(service)

    summary = service.train(VIEW, "geometry")

    report = summary["training_screening"]
    accepted_ids = {sample.id for sample in candidates} - {bad_train["id"]}
    assert report["candidate_images"] == len(candidates)
    assert report["accepted_images"] == summary["training_images"] == len(accepted_ids)
    assert report["excluded_images"] == 1
    assert set(report["accepted_sample_ids"]) == accepted_ids
    excluded = report["excluded"][0]
    assert excluded["sample_id"] == bad_train["id"]
    assert excluded["filename"] == Path(bad_train["path"]).name
    assert excluded["path"] == bad_train["path"] and excluded["part_id"] == train_part
    assert excluded["view"] == VIEW and excluded["stage"] == "quality"
    assert "insufficient_focus" in excluded["reasons"]
    assert excluded["metrics"]["sharpness"] < service.config["quality"]["min_sharpness"]
    assert dataset_bytes(service) == immutable
    metadata = ArtifactStore.verify(Path(summary["artifact_path"]))
    assert_original_roster_and_actual_training(metadata, split, accepted_ids)
    assert metadata["training_screening"]["excluded"][0]["sample_id"] == bad_train["id"]
    assert_screening_persisted(service, report)
    assert bad_validation["id"] in {row["sample_id"] for row in summary["thresholds"]["invalid_validation_captures"]}
    test_observation = next(row for row in summary["report"]["records"] if row["sample_id"] == bad_test["id"])
    assert test_observation["verdict"] == "RECHECK" and test_observation["error"] is None
    assert "insufficient_focus" in test_observation["reasons"]


def test_registration_rejection_uses_other_training_images_and_keeps_rejected_capture_out_of_reference(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    import_housings(service)
    split = current_split(service)
    candidates = good_training(split)
    checker, locator = QualityChecker(service.config["quality"]), HousingLocator()
    def sharpness(sample):
        image = read_image(sample.path)
        return checker.check(image, locator.detect(image).mask).metrics["sharpness"]
    # Exercise the case where the initially sharpest template is unsuitable.
    rejected = max(candidates, key=sharpness)
    rejected_image = read_image(rejected.path)
    immutable = dataset_bytes(service)
    original_register = LandmarkRegistrar.register
    original_fit = GeometryProfile.fit
    fit_sizes = []
    def register(registrar, image):
        if np.array_equal(image, rejected_image):
            identity = np.eye(3)
            return Registration(image.copy(), 0., identity, identity,
                                ["Controlled unavailable landmark consensus"], valid=False)
        return original_register(registrar, image)
    def fit(profile, images, *args, **kwargs):
        fit_sizes.append(len(images))
        return original_fit(profile, images, *args, **kwargs)
    monkeypatch.setattr(LandmarkRegistrar, "register", register)
    monkeypatch.setattr(GeometryProfile, "fit", fit)

    summary = service.train(VIEW, "geometry")

    report = summary["training_screening"]
    accepted_ids = {sample.id for sample in candidates} - {rejected.id}
    assert set(report["accepted_sample_ids"]) == accepted_ids
    assert report["excluded_images"] == 1
    excluded = report["excluded"][0]
    assert excluded["sample_id"] == rejected.id and excluded["stage"] == "registration"
    assert "Controlled unavailable landmark consensus" in excluded["reasons"]
    assert fit_sizes == [len(accepted_ids)]
    assert summary["training_images"] == len(accepted_ids)
    metadata = ArtifactStore.verify(Path(summary["artifact_path"]))
    assert_original_roster_and_actual_training(metadata, split, accepted_ids)
    template = read_image(Path(summary["artifact_path"]) / "template.png")
    assert not np.array_equal(template, rejected_image)
    assert dataset_bytes(service) == immutable
    assert_screening_persisted(service, report)


def test_all_training_captures_rejected_persists_report_and_preserves_active_model(tmp_path):
    service = make_service(tmp_path)
    import_housings(service)
    previous = service.train(VIEW, "geometry")
    active = service.artifacts.active(VIEW, "geometry")
    manifest = (active / "manifest.json").read_bytes()
    immutable = dataset_bytes(service)
    split = current_split(service)
    candidates = good_training(split)
    service.config["quality"]["min_sharpness"] = 1e12

    with pytest.raises(ValueError) as captured:
        service.train(VIEW, "geometry")

    report = service.training_screening(VIEW)
    assert report["candidate_images"] == report["excluded_images"] == len(candidates)
    assert report["accepted_images"] == 0 and report["accepted_sample_ids"] == []
    assert {record["sample_id"] for record in report["excluded"]} == {sample.id for sample in candidates}
    assert all("insufficient_focus" in record["reasons"] for record in report["excluded"])
    message = str(captured.value)
    assert "insufficient_focus" in message and str(report["report_path"]) in message
    assert "No usable GOOD training captures remain" in message
    assert f"{len(candidates)} of {len(candidates)}" in message
    assert_screening_persisted(service, report)
    assert service.artifacts.active(VIEW, "geometry") == active
    assert (active / "manifest.json").read_bytes() == manifest
    assert Path(previous["artifact_path"]) == active
    assert dataset_bytes(service) == immutable
    restarted = NeuroIrisService(service.config)
    assert restarted.training_screening(VIEW)["report_path"] == report["report_path"]


def test_empty_inspection_roi_excludes_captures_without_modifying_dataset(tmp_path):
    service = make_service(tmp_path)
    import_housings(service)
    split = current_split(service)
    candidates = good_training(split)
    service.save_rois(VIEW, [{"shape": "rectangle", "points": [[0., 0.], [1., 1.]],
                              "exclusion": True}])
    immutable = dataset_bytes(service)

    with pytest.raises(ValueError, match="No usable GOOD training captures"):
        service.train(VIEW, "geometry")

    report = service.training_screening(VIEW)
    assert report["accepted_images"] == 0
    assert report["excluded_images"] == report["candidate_images"] == len(candidates)
    assert {record["sample_id"] for record in report["excluded"]} == {sample.id for sample in candidates}
    assert all(record["stage"] == "roi" for record in report["excluded"])
    assert all("inspection_roi_has_too_few_part_pixels" in record["reasons"] for record in report["excluded"])
    assert all(record["metrics"]["roi_pixels"] == 0 for record in report["excluded"])
    assert dataset_bytes(service) == immutable
    assert_screening_persisted(service, report)


def test_post_inventory_source_change_remains_integrity_error_not_quality_exclusion(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    import_housings(service)
    target = good_training(current_split(service))[0]
    original_bytes = target.path.read_bytes()
    original_sidecar = target.path.with_suffix(target.path.suffix + ".json").read_bytes()
    original_inventory = service.dataset.inventory
    mutated = False
    def inventory(*args, **kwargs):
        nonlocal mutated
        samples = original_inventory(*args, **kwargs)
        if not mutated:
            # A valid image plus trailing bytes still decodes; its provenance has changed.
            target.path.write_bytes(original_bytes + b"post-inventory-change")
            mutated = True
        return samples
    monkeypatch.setattr(service.dataset, "inventory", inventory)

    with pytest.raises(DatasetIntegrityError, match="changed after inventory"):
        service.train(VIEW, "geometry")

    report = service.training_screening(VIEW)
    assert report["status"] == "failed"
    assert report["excluded_images"] == 0 and report["excluded"] == []
    assert "changed after inventory" in report["error"]
    assert target.path.read_bytes() == original_bytes + b"post-inventory-change"
    assert target.path.with_suffix(target.path.suffix + ".json").read_bytes() == original_sidecar
    assert_screening_persisted(service, report)
