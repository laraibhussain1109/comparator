"""Checkpoint/cache filesystem regression tests; no neural accuracy assertions."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from src.models import EfficientADBackend, ModelSetupError
from src.models.efficientad import ANOMALIB_VERSION, ARTIFACT_VERSION, SCORE_UNITS
from src.models import weights
from src.utils.filesystem import filesystem_path


def deep_unicode_directory(tmp_path):
    # Exceed legacy MAX_PATH without exceeding any single component's limit.
    directory = tmp_path.joinpath(*["NeuroIris aluminium inspection housing results " + str(i) for i in range(5)],
                                 "pièce contrôlée 铝壳")
    Path(filesystem_path(directory)).mkdir(parents=True)
    assert len(str(directory)) > 260
    return directory


def model_metadata():
    return {"artifact_version": ARTIFACT_VERSION, "anomalib_version": ANOMALIB_VERSION,
            "score_units": SCORE_UNITS, "per_image_normalization": False,
            "teacher": {"sha256": "a" * 64}, "optimizer_steps": 1,
            "input_size": 512, "model_size": "small"}


class StateOnlyModel:
    """A serialization fixture; never used to generate neural predictions."""
    def __init__(self, state=None):
        self.state = state or {}

    def state_dict(self):
        return self.state

    def load_state_dict(self, state, strict):
        assert strict is True
        self.state = state

    def eval(self):
        return self


class StreamSerializer:
    """Serialization double guards the Python file-object boundary without torch."""
    def __init__(self):
        self.calls = []

    def save(self, state, file):
        assert isinstance(file, io.BufferedIOBase), "Native filename writer must not be used"
        self.calls.append("save")
        file.write(b"tensor serialization software fixture")

    def load(self, file, *, map_location, weights_only):
        assert isinstance(file, io.BufferedIOBase), "Native filename reader must not be used"
        assert map_location == "cpu" and weights_only is True
        assert file.read() == b"tensor serialization software fixture"
        self.calls.append("load")
        return {}


def test_checkpoint_uses_python_streams_for_long_unicode_paths(tmp_path, monkeypatch):
    directory = deep_unicode_directory(tmp_path)
    serializer = StreamSerializer()
    backend = EfficientADBackend()
    backend.model = StateOnlyModel()
    backend._torch = serializer
    backend._trained = True
    backend.metadata = model_metadata()
    backend.save(directory)
    restored = EfficientADBackend()
    monkeypatch.setattr(restored, "_build", lambda: (
        setattr(restored, "model", StateOnlyModel()), setattr(restored, "_torch", serializer)))
    restored.load(directory)
    assert restored.trained and serializer.calls == ["save", "load"]
    with open(filesystem_path(directory / "model.json"), encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["checkpoint_sha256"] == weights.sha256_file(directory / "efficientad.pt")
    assert restored.metadata == metadata
    assert not Path(filesystem_path(directory / ".ea.pt.tmp")).exists()
    assert not Path(filesystem_path(directory / ".ea.json.tmp")).exists()


def test_teacher_cache_and_natural_images_at_long_unicode_paths(tmp_path, monkeypatch):
    directory = deep_unicode_directory(tmp_path)
    archive = directory / "efficientad_pretrained_weights.zip"
    # Explicit cache software fixture, not real pretrained weights.
    with open(filesystem_path(archive), "wb") as handle, zipfile.ZipFile(handle, "w") as source:
        source.writestr("efficientad_pretrained_weights/pretrained_teacher_small.pth", b"cache software fixture")
    monkeypatch.setattr(weights, "TEACHER_SHA256", weights.sha256_file(archive))
    target, provenance = weights.teacher_weights({"teacher_cache": str(directory), "allow_download": False})
    assert provenance["sha256"] == hashlib.sha256(b"cache software fixture").hexdigest()
    Path(filesystem_path(archive)).unlink()
    assert weights.teacher_weights({"teacher_cache": str(directory), "allow_download": False}) == (target, provenance)
    natural = directory / "photograph.jpg"
    with open(filesystem_path(natural), "wb") as handle:
        handle.write(b"directory traversal fixture only; no inference")
    assert weights.natural_image_paths({"imagenet_dir": str(directory)}) == [natural]


def test_verified_download_long_path_checks_digest_and_cleans_temporary(tmp_path, monkeypatch):
    directory = deep_unicode_directory(tmp_path)
    payload = b"HTTPS download software fixture"
    monkeypatch.setattr(weights.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    target = directory / "official-asset.zip"
    assert weights.verified_download("https://example.org/asset.zip", target, hashlib.sha256(payload).hexdigest()) == target
    assert weights.sha256_file(target) == hashlib.sha256(payload).hexdigest()
    rejected = directory / "invalid-asset.zip"
    with pytest.raises(ModelSetupError, match="SHA256 mismatch"):
        weights.verified_download("https://example.org/asset.zip", rejected, "0" * 64)
    assert not Path(filesystem_path(rejected)).exists()
    assert not list(Path(filesystem_path(directory)).glob(".dl-*.tmp"))


def test_real_tensor_checkpoint_uses_streams_and_reloads_safely(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    directory = deep_unicode_directory(tmp_path)
    calls = []

    def save(state, file):
        assert isinstance(file, io.BufferedIOBase)
        calls.append("save")
        torch.save(state, file)

    def load(file, *, map_location, weights_only):
        assert isinstance(file, io.BufferedIOBase)
        assert weights_only is True
        calls.append("load")
        return torch.load(file, map_location=map_location, weights_only=weights_only)

    serializer = SimpleNamespace(save=save, load=load)
    original = EfficientADBackend()
    original._torch = serializer
    original.model = StateOnlyModel({"weights": torch.arange(12, dtype=torch.float32).reshape(3, 4)})
    original._trained = True
    original.metadata = model_metadata()
    original.save(directory)
    restored = EfficientADBackend()
    monkeypatch.setattr(restored, "_build", lambda: (
        setattr(restored, "model", StateOnlyModel()), setattr(restored, "_torch", serializer)))
    restored.load(directory)
    assert calls == ["save", "load"]
    torch.testing.assert_close(original.model.state["weights"], restored.model.state["weights"])
