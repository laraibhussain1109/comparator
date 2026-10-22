"""Missing natural photographs are provisioned without a UI/settings gate."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import tarfile

import pytest

from src.models import weights


PHOTO = b"natural photograph archive software fixture; no inference"


def archive_bytes(entries):
    """Create tiny archive fixtures, never production training images."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, payload, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.SYMTYPE:
                member.linkname = "../../outside.jpg"
            elif kind == tarfile.REGTYPE:
                member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload) if kind == tarfile.REGTYPE else None)
    return stream.getvalue()


def provide_archive(monkeypatch, payload):
    requests = []
    monkeypatch.setattr(weights, "IMAGENETTE_SHA256", hashlib.sha256(payload).hexdigest())

    def urlopen(request, **kwargs):
        requests.append(request.full_url)
        return io.BytesIO(payload)

    monkeypatch.setattr(weights.urllib.request, "urlopen", urlopen)
    return requests


@pytest.mark.parametrize("legacy_flag", [None, False, True])
def test_missing_photos_download_without_checkbox_even_with_saved_false(tmp_path, monkeypatch, legacy_flag):
    payload = archive_bytes([("imagenette2/train/n001/photo.jpg", PHOTO, tarfile.REGTYPE)])
    requests = provide_archive(monkeypatch, payload)
    root = tmp_path / "custom natural photographs"
    teacher = tmp_path / "separate asset cache" / "teacher"
    settings = {"imagenet_dir": str(root), "teacher_cache": str(teacher)}
    if legacy_flag is not None:
        settings["download_imagenette"] = legacy_flag
    messages = []

    paths = weights.natural_image_paths(settings, progress=messages.append)

    assert paths == [root / "n001" / "photo.jpg"]
    assert paths[0].read_bytes() == PHOTO
    assert requests == [weights.IMAGENETTE_URL]
    assert (teacher.parent / "imagenette2.tgz").read_bytes() == payload
    assert any("Extracting" in message for message in messages)
    assert any("ready" in message for message in messages)


def test_default_directories_are_created_automatically(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    payload = archive_bytes([("imagenette2/train/n001/photo.jpg", PHOTO, tarfile.REGTYPE)])
    requests = provide_archive(monkeypatch, payload)

    paths = weights.natural_image_paths({})

    assert paths == [Path("artifacts/pretrained/imagenette2/train/n001/photo.jpg")]
    assert paths[0].read_bytes() == PHOTO
    assert Path("artifacts/pretrained/imagenette2.tgz").read_bytes() == payload
    assert requests == [weights.IMAGENETTE_URL]


def test_existing_natural_photographs_are_reused_without_network(tmp_path, monkeypatch):
    root = tmp_path / "existing natural photographs"
    photograph = root / "nested class" / "existing.jpg"
    photograph.parent.mkdir(parents=True)
    photograph.write_bytes(PHOTO)

    def no_network(*args, **kwargs):
        pytest.fail("An existing natural-image folder must remain usable offline")

    monkeypatch.setattr(weights.urllib.request, "urlopen", no_network)

    assert weights.natural_image_paths({"imagenet_dir": str(root), "download_imagenette": False}) == [photograph]
    assert photograph.read_bytes() == PHOTO


def test_custom_directories_reuse_downloaded_photos_on_later_training(tmp_path, monkeypatch):
    payload = archive_bytes([("imagenette2/train/n001/photo.jpg", PHOTO, tarfile.REGTYPE)])
    requests = provide_archive(monkeypatch, payload)
    root = tmp_path / "user selected natural images"
    teacher = tmp_path / "other cache" / "teacher"
    settings = {"imagenet_dir": str(root), "teacher_cache": str(teacher)}
    first = weights.natural_image_paths(settings)
    # Delete the archive to prove the second call uses the photographs rather
    # than re-downloading or re-extracting an unrelated cache directory.
    (teacher.parent / "imagenette2.tgz").unlink()

    second = weights.natural_image_paths(settings)

    assert second == first == [root / "n001" / "photo.jpg"]
    assert requests == [weights.IMAGENETTE_URL]


def test_verified_cached_archive_extracts_automatically_without_network(tmp_path, monkeypatch):
    payload = archive_bytes([("imagenette2/train/n001/photo.jpg", PHOTO, tarfile.REGTYPE)])
    monkeypatch.setattr(weights, "IMAGENETTE_SHA256", hashlib.sha256(payload).hexdigest())
    root = tmp_path / "natural images"
    teacher = tmp_path / "cache" / "teacher"
    teacher.parent.mkdir()
    archive = teacher.parent / "imagenette2.tgz"
    archive.write_bytes(payload)

    def no_network(*args, **kwargs):
        pytest.fail("A checksum-verified browser-downloaded archive must work offline")

    monkeypatch.setattr(weights.urllib.request, "urlopen", no_network)

    paths = weights.natural_image_paths({"imagenet_dir": str(root), "teacher_cache": str(teacher),
                                        "download_imagenette": False})

    assert paths == [root / "n001" / "photo.jpg"]
    assert paths[0].read_bytes() == PHOTO
    assert archive.read_bytes() == payload


def test_extraction_accepts_only_regular_training_images_inside_expected_tree(tmp_path, monkeypatch):
    entries = [
        ("imagenette2/train/n001/photo.jpg", PHOTO, tarfile.REGTYPE),
        ("imagenette2/val/n001/validation.jpg", PHOTO, tarfile.REGTYPE),
        ("imagenette2/train/n001/readme.txt", b"description", tarfile.REGTYPE),
        ("imagenette2/train/../outside.jpg", PHOTO, tarfile.REGTYPE),
        ("imagenette2/train/n001/link.jpg", b"", tarfile.SYMTYPE),
        ("other/train/n001/other.jpg", PHOTO, tarfile.REGTYPE),
    ]
    requests = provide_archive(monkeypatch, archive_bytes(entries))
    root = tmp_path / "natural images"
    settings = {"imagenet_dir": str(root), "teacher_cache": str(tmp_path / "cache" / "teacher")}

    paths = weights.natural_image_paths(settings)

    assert paths == [root / "n001" / "photo.jpg"]
    assert sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file()) == [Path("n001/photo.jpg")]
    assert not (tmp_path / "outside.jpg").exists()
    assert requests == [weights.IMAGENETTE_URL]


def test_archive_without_usable_training_images_fails_honestly(tmp_path, monkeypatch):
    payload = archive_bytes([("imagenette2/val/n001/photo.jpg", PHOTO, tarfile.REGTYPE)])
    requests = provide_archive(monkeypatch, payload)
    root = tmp_path / "natural images"

    with pytest.raises(weights.ModelSetupError, match="no usable training photographs"):
        weights.natural_image_paths({"imagenet_dir": str(root), "teacher_cache": str(tmp_path / "cache" / "teacher")})

    assert not list(root.rglob("*.jpg"))
    assert requests == [weights.IMAGENETTE_URL]
