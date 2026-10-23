"""Invalid held-out captures stop fitting without weakening capture gates."""
from pathlib import Path

import cv2
import numpy as np
import pytest

import src.service as service_module
from src.models import EfficientADBackend
from src.vision import LandmarkRegistrar, Registration
from test_smoke_training import import_housings, make_service
from test_training_screening import VIEW, current_split, dataset_bytes


@pytest.mark.parametrize("failure", ["quality", "registration", "resolution"])
def test_invalid_reserved_good_captures_fail_before_neural_fit_with_actionable_report(
        tmp_path, monkeypatch, failure):
    service = make_service(tmp_path)
    import_housings(service)
    previous = service.train(VIEW, "geometry")
    active = service.artifacts.active(VIEW, "geometry")
    manifest = (active / "manifest.json").read_bytes()
    split = current_split(service)
    reserved = [sample for sample in split.validation if sample.view == VIEW and sample.label == 0]
    originals = dataset_bytes(service)
    original_read = service_module.read_image
    validation_paths = {sample.path for sample in reserved}
    validation_images = [original_read(sample.path) for sample in reserved]
    fit_calls = []

    def read(path):
        image = original_read(path)
        if Path(path) in validation_paths:
            if failure == "quality":
                return cv2.GaussianBlur(image, (25, 25), 5)
            if failure == "resolution":
                return cv2.resize(image, None, fx=.5, fy=.5)
        return image

    original_register = LandmarkRegistrar.register

    def register(registrar, image):
        if failure == "registration" and any(np.array_equal(image, reference)
                                            for reference in validation_images):
            identity = np.eye(3)
            return Registration(image.copy(), 0., identity, identity,
                                ["Controlled unavailable validation landmarks"], valid=False)
        return original_register(registrar, image)

    monkeypatch.setattr(service_module, "read_image", read)
    monkeypatch.setattr(LandmarkRegistrar, "register", register)
    monkeypatch.setattr(EfficientADBackend, "fit", lambda *args, **kwargs: fit_calls.append(True))

    with pytest.raises(ValueError, match="No independent GOOD validation captures") as captured:
        service.train(VIEW, "efficientad")

    report = service.training_screening(VIEW)
    screening = report["validation_screening"]
    assert fit_calls == []
    assert screening["phase"] == "preflight" and screening["status"] == "failed"
    assert screening["good_accepted_images"] == 0
    assert screening["good_candidate_images"] == len(reserved)
    assert set(screening["reserved_good_part_ids"]) == {sample.part_id for sample in reserved}
    invalid = report["invalid_validation_captures"]
    assert {row["sample_id"] for row in invalid} == {sample.id for sample in reserved}
    assert all(row["stage"] == failure for row in invalid)
    assert all(row["reasons"] and row["path"] for row in invalid)
    assert all(sample.part_id in str(captured.value) for sample in reserved)
    assert all(reason in str(captured.value) for row in invalid for reason in row["reasons"])
    assert report["report_path"] in str(captured.value)
    if failure == "quality":
        assert all(row["metrics"]["sharpness"] < service.config["quality"]["min_sharpness"]
                   for row in invalid)
    if failure == "registration":
        assert all(row["metrics"]["registration_confidence"] == 0 for row in invalid)
    assert dataset_bytes(service) == originals
    assert service.artifacts.active(VIEW, "geometry") == Path(previous["artifact_path"])
    assert (active / "manifest.json").read_bytes() == manifest
    assert current_split(service).manifest() == split.manifest()

