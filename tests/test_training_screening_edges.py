"""Regression checks for screening reference choice and audit-write failure.

These software housings exercise workflow behavior, not industrial accuracy.
"""
from pathlib import Path

import cv2
import pytest

from src.acquisition import QualityChecker
from src.artifacts import ArtifactStore
from src.service import NeuroIrisService
from src.utils.image_utils import read_image
from src.vision import HousingLocator
from test_smoke_training import import_housings, make_service
from test_training_screening import VIEW, current_split, dataset_bytes, good_training
from test_vision import housing


def test_sharper_singleton_thumbnail_cannot_replace_representative_native_master(tmp_path):
    service = make_service(tmp_path)
    import_housings(service)
    native_samples = good_training(current_split(service))
    assert len(native_samples) > 1
    thumbnail = cv2.resize(housing(True), None, fx=.5, fy=.5,
                           interpolation=cv2.INTER_NEAREST)
    thumbnail_record = service.capture_sample(thumbnail, VIEW, "GOOD", native_samples[0].part_id,
                                              "software-fixture-thumbnail")
    checker, locator = QualityChecker(service.config["quality"]), HousingLocator()
    thumbnail_quality = checker.check(thumbnail, locator.detect(thumbnail).mask)
    native_sharpness = [checker.check(image, locator.detect(image).mask).metrics["sharpness"]
                        for image in (read_image(sample.path) for sample in native_samples)]
    assert thumbnail_quality.valid
    assert thumbnail_quality.metrics["sharpness"] > max(native_sharpness)
    assert thumbnail_record["id"] in {sample.id for sample in good_training(current_split(service))}
    original_thumbnail = Path(thumbnail_record["path"]).read_bytes()

    summary = service.train(VIEW, "geometry")

    screening = summary["training_screening"]
    assert screening["template_sample_id"] in {sample.id for sample in native_samples}
    assert screening["template_sample_id"] != thumbnail_record["id"]
    assert {sample.id for sample in native_samples} <= set(screening["accepted_sample_ids"])
    assert not {sample.id for sample in native_samples} & {
        row["sample_id"] for row in screening["excluded"]}
    template = read_image(Path(summary["artifact_path"]) / "template.png")
    assert template.shape == housing(True).shape
    assert Path(thumbnail_record["path"]).read_bytes() == original_thumbnail
    metadata = ArtifactStore.verify(Path(summary["artifact_path"]))
    assert metadata["training_screening"]["template_sample_id"] == screening["template_sample_id"]


def test_external_completion_audit_write_failure_does_not_report_published_model_as_failed(tmp_path,
                                                                                         monkeypatch):
    service = make_service(tmp_path)
    import_housings(service)
    previous = service.train(VIEW, "geometry")
    previous_active = service.artifacts.active(VIEW, "geometry")
    original_save = service._save_training_screening

    def save(report):
        if report["status"] == "completed":
            raise OSError("Controlled external completion-audit write failure")
        return original_save(report)

    monkeypatch.setattr(service, "_save_training_screening", save)

    summary = service.train(VIEW, "geometry")

    active = service.artifacts.active(VIEW, "geometry")
    assert active == Path(summary["artifact_path"]) and active != previous_active
    assert previous_active == Path(previous["artifact_path"])
    assert ArtifactStore.verify(previous_active)["version"] == previous["version"]
    metadata = ArtifactStore.verify(active)
    assert metadata["training_screening"]["status"] == "completed"
    assert summary["training_screening"]["status"] == "completed"
    fallback = Path(summary["training_screening"]["report_path"])
    assert fallback == active / "training_screening.json" and fallback.is_file()
    assert any("audit" in str(warning).lower() and "write" in str(warning).lower()
               for warning in summary["warnings"])
    recovered = service.training_screening(VIEW)
    assert recovered["status"] == "completed"
    assert recovered["run_id"] == summary["training_screening"]["run_id"]
    assert Path(recovered["report_path"]) == fallback
    restarted = NeuroIrisService(service.config)
    assert restarted.training_screening(VIEW)["run_id"] == recovered["run_id"]


def test_nonempty_roi_below_geometry_area_minimum_is_excluded_during_screening(tmp_path):
    service = make_service(tmp_path)
    import_housings(service)
    candidates = good_training(current_split(service))
    service.save_rois(VIEW, [{"shape": "rectangle", "points": [[.49, .49], [.51, .51]],
                             "exclusion": False}])
    original_files = dataset_bytes(service)

    with pytest.raises(ValueError, match="No usable GOOD training captures remain"):
        service.train(VIEW, "geometry")

    report = service.training_screening(VIEW)
    assert report["status"] == "failed"
    assert report["accepted_images"] == 0
    assert report["excluded_images"] == report["candidate_images"] == len(candidates)
    assert all(row["stage"] == "roi" for row in report["excluded"])
    assert all(30 <= row["metrics"]["roi_pixels"] < row["metrics"]["minimum_roi_pixels"]
               for row in report["excluded"])
    assert dataset_bytes(service) == original_files
