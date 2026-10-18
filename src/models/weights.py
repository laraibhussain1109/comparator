"""Verified, reusable EfficientAD pretrained assets; no unsafe pickle loading.

The archive hashes below are published in Anomalib v2.2.0's EfficientAD
lightning_model.py. A recorded local digest is an integrity check, not a claim
that an arbitrary user supplied checkpoint has an authenticated publisher.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

TEACHER_URL = "https://github.com/open-edge-platform/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip"
TEACHER_SHA256 = "c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a"
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
IMAGENETTE_SHA256 = "6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ModelSetupError(RuntimeError):
    """A real neural model prerequisite is missing or invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_download(url: str, path: Path, expected_sha256: str) -> Path:
    """Download with normal TLS verification and verify a published SHA256."""
    if path.is_file() and sha256_file(path) == expected_sha256:
        return path
    if not url.startswith("https://"):
        raise ModelSetupError("Pretrained assets must use verified HTTPS URLs.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "NeuroIris/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
        actual = sha256_file(temporary)
        if actual != expected_sha256:
            raise ModelSetupError(f"SHA256 mismatch for {path.name}; downloaded asset will not be used.")
        os.replace(temporary, path)
        return path
    except (OSError, urllib.error.URLError) as error:
        raise ModelSetupError(
            f"Could not download {url}: {error}. Cache the official asset at {path} "
            "and retry; no untrained teacher will be substituted."
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def teacher_weights(settings: dict) -> tuple[Path, dict]:
    """Return a verified small/medium PDN teacher and explicit provenance."""
    custom = settings.get("teacher_path")
    if custom:
        path = Path(custom).expanduser()
        expected = str(settings.get("teacher_sha256", "")).lower()
        if not path.is_file():
            raise ModelSetupError(f"Pretrained teacher is missing: {path}")
        if len(expected) != 64 or sha256_file(path) != expected:
            raise ModelSetupError("A custom teacher_path requires its matching teacher_sha256.")
        return path, {"source": "user_supplied", "sha256": expected, "authenticity": "user_supplied_digest"}
    cache = Path(settings.get("teacher_cache") or "artifacts/pretrained/efficientad").expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "efficientad_pretrained_weights.zip"
    name = f"pretrained_teacher_{settings.get('model_size', 'small')}.pth"
    target = cache / name
    manifest_path = target.with_suffix(".integrity.json")
    if target.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("archive_sha256") == TEACHER_SHA256
                    and manifest.get("source") == TEACHER_URL
                    and sha256_file(target) == manifest.get("sha256")):
                return target, manifest
        except (OSError, ValueError):
            pass
    if not archive.is_file() or sha256_file(archive) != TEACHER_SHA256:
        if not settings.get("allow_download", True):
            raise ModelSetupError(
                f"Verified EfficientAD teacher unavailable. Enable allow_download or cache "
                f"the official archive at {archive}. Expected SHA256: {TEACHER_SHA256}"
            )
        verified_download(TEACHER_URL, archive, TEACHER_SHA256)
    with zipfile.ZipFile(archive) as source:
        matches = [entry for entry in source.infolist() if Path(entry.filename).name == name and not entry.is_dir()]
        if len(matches) != 1:
            raise ModelSetupError(f"Official EfficientAD archive does not contain exactly one {name}.")
        if matches[0].file_size > 100 * 1024 * 1024:
            raise ModelSetupError("Unexpected teacher checkpoint size.")
        # Copy one known basename rather than extracting archive paths.
        with source.open(matches[0]) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    manifest = {"source": TEACHER_URL, "archive_sha256": TEACHER_SHA256,
                "sha256": sha256_file(target), "authenticity": "published_anomalib_2.2.0_archive_sha256"}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target, manifest


def natural_image_paths(settings: dict) -> list[Path]:
    """Find genuine natural images for EfficientAD's required penalty term.

    This never reuses aluminium GOOD images as an ImageNet substitute. The user
    may provide an ImageNet/ImageNette folder. Downloading the official ~1.5 GB
    ImageNette archive is explicit because it is a substantial prerequisite.
    """
    root = Path(settings.get("imagenet_dir") or "artifacts/pretrained/imagenette2/train").expanduser()
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if root.is_dir() else []
    if paths:
        return paths
    if not settings.get("download_imagenette", False):
        raise ModelSetupError(
            f"EfficientAD requires a natural-image penalty dataset, absent at {root}. "
            "Set imagenet_dir to genuine ImageNet/ImageNette training photographs, "
            "or enable download_imagenette (official archive about 1.5 GB). "
            "GOOD housing images are not a replacement for this dataset."
        )
    cache = Path(settings.get("teacher_cache") or "artifacts/pretrained/efficientad").parent
    archive = verified_download(IMAGENETTE_URL, cache / "imagenette2.tgz", IMAGENETTE_SHA256)
    extracted = cache / "imagenette2"
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            parts = Path(member.name).parts
            # Only regular training images inside the expected dataset directory.
            if (not member.isfile() or len(parts) < 4 or parts[0:2] != ("imagenette2", "train")
                    or ".." in parts or Path(member.name).suffix.lower() not in IMAGE_SUFFIXES):
                continue
            output = cache.joinpath(*parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            stream = source.extractfile(member)
            if stream is not None:
                with stream, output.open("wb") as target:
                    shutil.copyfileobj(stream, target)
    paths = sorted(p for p in (extracted / "train").rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ModelSetupError("The verified ImageNette archive contains no usable training photographs.")
    return paths
