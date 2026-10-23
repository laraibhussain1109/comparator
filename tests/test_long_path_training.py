"""Real geometry workflow under long Unicode paths, runnable on Windows/POSIX.

Synthetic housing fixtures exercise filesystem behavior, not hardware accuracy.
The host test does not claim that Windows itself was exercised on a POSIX host.
"""
from hashlib import sha256
import json

import cv2

from src.artifacts import ArtifactStore
from src.data import split_by_part
from src.service import NeuroIrisService
from src.utils.filesystem import filesystem_path
from src.utils.image_utils import read_image
from test_smoke_training import import_housings, make_service
from test_vision import housing


def file_hashes(root):
    root = filesystem_path(root)
    return {path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_geometry_training_reload_and_evaluation_with_long_unicode_base(tmp_path):
    base = tmp_path
    # Components stay within per-component limits; total path exceeds MAX_PATH.
    while len(str(base)) < 285:
        base /= "inspection-workspace_铝件_परीक्षण_" + str(len(base.parts))
    base = filesystem_path(base)
    base.mkdir(parents=True, exist_ok=True)
    assert len(str(base)) > 260
    service = make_service(base)
    import_housings(service)
    split = split_by_part(service.dataset.inventory(), seed=service.config["runtime"]["random_seed"])
    train_part = next(sample.part_id for sample in split.train if sample.view == "ribbed_face" and sample.label == 0)

    source = filesystem_path(base / "incoming_originals" / "原始_capture_धुंधला.png")
    source.parent.mkdir(parents=True, exist_ok=True)
    blurry = cv2.GaussianBlur(housing(True), (25, 25), 5)
    success, encoded = cv2.imencode(".png", blurry)
    assert success
    original_bytes = encoded.tobytes()
    source.write_bytes(original_bytes)
    imported = service.import_sample(source, "ribbed_face", "GOOD", train_part, "long-path-fixture")
    imported_path = filesystem_path(imported["path"])
    assert imported_path.read_bytes() == original_bytes
    assert read_image(imported_path).shape == blurry.shape
    before = file_hashes(service.dataset.root)

    summary = service.train("ribbed_face", "geometry")

    directory = filesystem_path(summary["artifact_path"])
    assert directory.is_dir() and len(str(directory)) > 260
    metadata = ArtifactStore.verify(directory)
    assert metadata["view"] == "ribbed_face" and metadata["backend"] == "geometry"
    assert read_image(directory / "template.png").shape == housing(True).shape
    assert summary["training_screening"]["excluded_images"] == 1
    assert summary["training_screening"]["excluded"][0]["sample_id"] == imported["id"]
    bundle_screening = json.loads((directory / "training_screening.json").read_text(encoding="utf-8"))
    assert bundle_screening["accepted_sample_ids"] == metadata["training_accepted_sample_ids"]
    assert source.read_bytes() == original_bytes
    assert imported_path.read_bytes() == original_bytes
    assert file_hashes(service.dataset.root) == before
    # Paths and metadata must remain valid JSON, including non-ASCII components.
    serialized = json.dumps(summary, ensure_ascii=False, allow_nan=False)
    assert json.loads(serialized)["version"] == summary["version"]

    restarted = NeuroIrisService(service.config)
    assert restarted.artifacts.active("ribbed_face", "geometry") == directory
    screening = restarted.training_screening("ribbed_face")
    assert filesystem_path(screening["report_path"]).is_file()
    assert screening["accepted_sample_ids"] == bundle_screening["accepted_sample_ids"]
    report = restarted.evaluate("ribbed_face", "geometry")
    assert report["model_version"] == summary["version"]
    assert report["records"] and all(record["error"] is None for record in report["records"])
    assert filesystem_path(report["report_path"]).is_dir()
    assert file_hashes(service.dataset.root) == before
