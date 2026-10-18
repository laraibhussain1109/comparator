"""Model software behavior tests; synthetic fixtures are not accuracy validation."""
from __future__ import annotations

from importlib import util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.models import EfficientADBackend, ModelSetupError
from src.models.registry import available_backends, create_backend
from src.models.weights import natural_image_paths, sha256_file, teacher_weights, verified_download


def test_geometry_import_does_not_import_torch():
    result = subprocess.run(
        [sys.executable, "-c", "import src.models; import sys; assert 'torch' not in sys.modules"],
        cwd=Path(__file__).parents[1], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_missing_model_fails_honestly(tmp_path):
    backend = EfficientADBackend()
    with pytest.raises(ModelSetupError, match="not trained/loaded"):
        backend.predict([np.zeros((256, 256, 3), np.uint8)])
    with pytest.raises(ModelSetupError, match="incomplete"):
        backend.load(tmp_path)
    with pytest.raises(ModelSetupError, match="No trained"):
        backend.save(tmp_path)


def test_unsupported_models_and_invalid_native_batch():
    assert available_backends()["dinomaly"]["available"] is False
    with pytest.raises(ModelSetupError, match="optional and unavailable"):
        create_backend("dinomaly")
    with pytest.raises(ValueError, match="train_batch_size=1"):
        EfficientADBackend({"train_batch_size": 2})
    with pytest.raises(ValueError, match="multiple of 64"):
        EfficientADBackend({"input_size": 128})
    with pytest.raises(ModelSetupError, match="not been validated"):
        EfficientADBackend({"mixed_precision": True})


def test_required_natural_images_not_silently_replaced(tmp_path):
    with pytest.raises(ModelSetupError, match="natural-image penalty dataset"):
        natural_image_paths({"imagenet_dir": str(tmp_path), "download_imagenette": False})
    image = np.zeros((256, 256, 3), np.uint8)
    with pytest.raises(ModelSetupError, match="natural-image penalty dataset"):
        EfficientADBackend({"imagenet_dir": str(tmp_path)}).fit([image], tmp_path / "output")
    assert not (tmp_path / "output" / "model.json").exists()


def test_teacher_download_offline_and_custom_integrity(tmp_path):
    with pytest.raises(ModelSetupError, match="teacher unavailable"):
        teacher_weights({"teacher_cache": str(tmp_path / "missing"), "allow_download": False})
    checkpoint = tmp_path / "untrusted.pth"
    checkpoint.write_bytes(b"test data - never used as a model")
    with pytest.raises(ModelSetupError, match="teacher_sha256"):
        teacher_weights({"teacher_path": str(checkpoint)})
    path, manifest = teacher_weights({"teacher_path": str(checkpoint), "teacher_sha256": sha256_file(checkpoint)})
    assert path == checkpoint
    assert manifest["authenticity"] == "user_supplied_digest"
    with pytest.raises(ModelSetupError, match="verified HTTPS"):
        verified_download("http://invalid.example/teacher.pth", tmp_path / "target", "0" * 64)


def test_checkpoint_hash_checked_before_torch_import(tmp_path):
    checkpoint = tmp_path / "efficientad.pt"
    checkpoint.write_bytes(b"corrupted checkpoint")
    info = {"artifact_version": 1, "anomalib_version": "2.2.0", "score_units": "raw_mean_squared_feature_distance_v1",
            "per_image_normalization": False, "teacher": {"sha256": "a" * 64}, "optimizer_steps": 1,
            "checkpoint_sha256": "0" * 64}
    (tmp_path / "model.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ModelSetupError, match="SHA256 integrity"):
        EfficientADBackend().load(tmp_path)


def test_invalid_images_fail_before_optional_dependencies(tmp_path):
    with pytest.raises(ValueError, match="uint8 BGR"):
        EfficientADBackend().fit([np.zeros((256, 256), np.uint8)], tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        EfficientADBackend().fit([], tmp_path)


@pytest.mark.skipif(not os.environ.get("NEUROIRIS_REAL_EFFICIENTAD_SMOKE"),
                    reason="Opt-in actual EfficientAD training smoke needs verified pretrained assets and natural photographs.")
def test_actual_efficientad_optimizer_and_checkpoint_roundtrip(tmp_path):
    """One real optimizer step on synthetic normal tiles; no industrial claims."""
    assert util.find_spec("anomalib") is not None
    import torch
    torch.set_num_threads(2)
    random = np.random.default_rng(24)
    # Synthetic software fixture, explicitly unrelated to real housing validation.
    good = [random.integers(50, 200, (256, 256, 3), dtype=np.uint8) for _ in range(2)]
    settings = {"device": "cpu", "input_size": 256, "max_steps": 1, "inference_batch_size": 1,
                "imagenet_dir": os.environ["NEUROIRIS_NATURAL_IMAGE_DIR"],
                "teacher_cache": os.environ["NEUROIRIS_TEACHER_CACHE"], "allow_download": False}
    progress = []
    model = EfficientADBackend(settings)
    independent_validation_fixture = random.integers(50, 200, (256, 256, 3), dtype=np.uint8)
    report = model.fit(good, tmp_path, validation_good=[independent_validation_fixture],
                       progress=lambda *args: progress.append(args))
    assert report["optimizer_steps"] == 1
    assert report["implementation"] == "anomalib.EfficientAdModel"
    assert report["teacher"]["authenticity"] == "published_anomalib_2.2.0_archive_sha256"
    assert np.isfinite(report["loss_history"][-1]["total"])
    assert report["training_device"] == "cpu"
    assert report["per_image_normalization"] is False
    first = model.predict([good[0]])[0]
    reloaded = EfficientADBackend({"device": "cpu"}).load(tmp_path)
    second = reloaded.predict([good[0]])[0]
    assert first.shape == good[0].shape[:2]
    assert first.dtype == np.float32
    assert np.isfinite(first).all() and np.ptp(first) > 0
    np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-6)
    # Original tile coordinates are preserved even when explicitly downsampled.
    large = np.repeat(np.repeat(good[0], 2, axis=0), 2, axis=1)
    assert reloaded.predict([large])[0].shape == (512, 512)
    assert progress[-1][0] == 100
