"""Verified, reusable EfficientAD pretrained assets; no unsafe pickle loading.

The archive hashes below are published in Anomalib v2.2.0's EfficientAD
lightning_model.py. A recorded local digest is an integrity check, not a claim
that an arbitrary user supplied checkpoint has an authenticated publisher.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
import ssl
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable
from uuid import uuid4

from src.utils.filesystem import filesystem_path

TEACHER_URL = "https://github.com/open-edge-platform/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip"
TEACHER_SHA256 = "c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a"
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
IMAGENETTE_SHA256 = "6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DOWNLOAD_ATTEMPTS = 5


class _DownloadInterrupted(Exception):
    """A transient response failure, distinct from a local file write failure."""


class ModelSetupError(RuntimeError):
    """A real neural model prerequisite is missing or invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(filesystem_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def https_context() -> ssl.SSLContext:
    """Use native certificate-chain validation, including Windows CryptoAPI.

    The per-request context applies to HTTPS redirects as well. It avoids a
    global SSL monkeypatch and keeps hostname/date/issuer verification enabled.
    """
    try:
        from truststore import SSLContext
    except ImportError as error:
        raise ModelSetupError(
            'Native HTTPS certificate support is missing. Install the updated application '
            'dependencies in the Python environment that runs NeuroIris: '
            'python -m pip install -r requirements.txt'
        ) from error
    return SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def verified_download(url: str, path: Path, expected_sha256: str, *,
                      progress: Callable[[str], None] | None = None) -> Path:
    """Retry interrupted HTTPS transfers and publish only checksum-verified files.

    A private temporary file retains received bytes between connection attempts.
    Range responses must start at the requested offset; a server ignoring Range
    restarts the file. Certificate failures, invalid ranges and local disk errors
    are not transient network errors. TLS verification is never disabled.
    """
    path = Path(path)
    emit = progress or (lambda _: None)
    if Path(filesystem_path(path)).is_file() and sha256_file(path) == expected_sha256:
        emit(f"Using verified cached {path.name}")
        return path
    if not url.startswith("https://"):
        raise ModelSetupError("Pretrained assets must use verified HTTPS URLs.")
    context = https_context()
    Path(filesystem_path(path.parent)).mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (".dl-" + uuid4().hex + ".tmp")
    validator = None
    network_errors = (OSError, urllib.error.URLError, http.client.HTTPException)
    try:
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            offset = Path(filesystem_path(temporary)).stat().st_size if Path(filesystem_path(temporary)).exists() else 0
            headers = {"User-Agent": "NeuroIris/1.0", "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
                if validator:
                    headers["If-Range"] = validator
            emit(f"{'Resuming' if offset else 'Downloading'} {path.name}: "
                 f"{offset / 1024 ** 2:.1f} MiB received (attempt {attempt}/{DOWNLOAD_ATTEMPTS})")
            request = urllib.request.Request(url, headers=headers)
            try:
                try:
                    response = urllib.request.urlopen(request, timeout=120, context=context)
                except urllib.error.HTTPError as error:
                    if error.code not in {408, 429} and not 500 <= error.code <= 599:
                        error.close()
                        raise ModelSetupError(f"Could not download {url}: HTTP {error.code} {error.reason}. "
                                              f"Save the official archive at {path} and retry.") from error
                    error.close()
                    raise _DownloadInterrupted(f"HTTP {error.code} {error.reason}") from error
                except network_errors as error:
                    _raise_transfer_error(error, url=url, path=path)
                with response:
                    if hasattr(response, "geturl") and not response.geturl().startswith("https://"):
                        raise ModelSetupError("Asset download redirected to an unverified HTTP URL.")
                    status = getattr(response, "status", None) or 200
                    response_headers = getattr(response, "headers", {})
                    length = response_headers.get("Content-Length")
                    try:
                        length = int(length) if length is not None else None
                        if length is not None and length < 0:
                            raise ValueError
                    except (TypeError, ValueError) as error:
                        raise ModelSetupError(f"Invalid Content-Length while downloading {path.name}.") from error
                    if status == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response_headers.get("Content-Range", ""))
                        if not match:
                            raise ModelSetupError(f"Invalid Content-Range while downloading {path.name}.")
                        start, end, total = map(int, match.groups())
                        if start != offset or not start <= end < total or (length is not None and length != end - start + 1):
                            raise ModelSetupError(f"Invalid Content-Range while resuming {path.name}; received bytes will not be appended.")
                        expected_response = end - start + 1
                    elif status == 200:
                        if offset:
                            emit(f"Server restarted {path.name}; restarting this transfer safely")
                        offset = 0
                        total = length
                        expected_response = length
                    else:
                        raise ModelSetupError(f"Unexpected HTTP {status} while downloading {path.name}.")
                    validator = response_headers.get("ETag") or response_headers.get("Last-Modified")
                    received = 0
                    reported_at, reported_bytes = time.monotonic(), offset
                    with open(filesystem_path(temporary), "ab" if offset else "wb") as target:
                        while True:
                            try:
                                chunk = response.read(1024 * 1024)
                            except network_errors as error:
                                _raise_transfer_error(error, url=url, path=path)
                            if not chunk:
                                break
                            if expected_response is not None and received + len(chunk) > expected_response:
                                raise ModelSetupError(f"Response exceeds its advertised size for {path.name}.")
                            target.write(chunk)
                            received += len(chunk)
                            current = offset + received
                            now = time.monotonic()
                            if current - reported_bytes >= 16 * 1024 ** 2 or now - reported_at >= 5:
                                emit(_download_status(path, current, total))
                                reported_at, reported_bytes = now, current
                    if expected_response is not None and received != expected_response:
                        raise _DownloadInterrupted(f"Connection ended after {received} of {expected_response} response bytes")
                    if total is not None and offset + received != total:
                        raise _DownloadInterrupted(f"Transfer incomplete: {offset + received} of {total} bytes")
                    emit(_download_status(path, offset + received, total))
                emit(f"Verifying SHA256 for {path.name}")
                if sha256_file(temporary) != expected_sha256:
                    raise ModelSetupError(f"SHA256 mismatch for {path.name}; downloaded asset will not be used.")
                os.replace(filesystem_path(temporary), filesystem_path(path))
                emit(f"Verified {path.name}; download complete")
                return path
            except _DownloadInterrupted as error:
                if attempt == DOWNLOAD_ATTEMPTS:
                    raise ModelSetupError(
                        f"Could not download {url} after {DOWNLOAD_ATTEMPTS} attempts: {error}. "
                        f"The HTTPS connection was interrupted. You can download the official archive "
                        f"in a browser, save it at {path}, then retry training; its SHA256 will be verified."
                    ) from error
                delay = 2 ** (attempt - 1)
                emit(f"Connection interrupted for {path.name}: {error}. "
                     f"Retrying in {delay}s; received bytes will be reused when the server supports resuming.")
                time.sleep(delay)
    except OSError as error:
        raise ModelSetupError(f"Could not write downloaded asset at {path}: {error}. "
                              "Check the folder permissions and available disk space.") from error
    finally:
        Path(filesystem_path(temporary)).unlink(missing_ok=True)


def _raise_transfer_error(error: Exception, *, url: str, path: Path) -> None:
    reason = getattr(error, "reason", error)
    if isinstance(reason, ssl.SSLCertVerificationError):
        raise ModelSetupError(
            f"HTTPS certificate verification failed while downloading {path.name} from {url}: {reason}. "
            "The server certificate chain was not trusted. Certificate and hostname checks remain enabled. "
            f"You can download this official archive in a browser and save it at {path}; "
            "the app will verify its SHA256 before use. Completed cached assets are retained."
        ) from error
    raise _DownloadInterrupted(str(error)) from error


def _download_status(path: Path, received: int, total: int | None) -> str:
    size = f"{received / 1024 ** 2:.1f} MiB"
    if total:
        size += f" / {total / 1024 ** 2:.1f} MiB ({received * 100 / total:.0f}%)"
    return f"Downloading {path.name}: {size}"


def teacher_weights(settings: dict, *, progress: Callable[[str], None] | None = None) -> tuple[Path, dict]:
    """Return a verified small/medium PDN teacher and explicit provenance."""
    custom = settings.get("teacher_path")
    if custom:
        path = Path(custom).expanduser()
        expected = str(settings.get("teacher_sha256", "")).lower()
        if not Path(filesystem_path(path)).is_file():
            raise ModelSetupError(f"Pretrained teacher is missing: {path}")
        if len(expected) != 64 or sha256_file(path) != expected:
            raise ModelSetupError("A custom teacher_path requires its matching teacher_sha256.")
        return path, {"source": "user_supplied", "sha256": expected, "authenticity": "user_supplied_digest"}
    cache = Path(settings.get("teacher_cache") or "artifacts/pretrained/efficientad").expanduser()
    Path(filesystem_path(cache)).mkdir(parents=True, exist_ok=True)
    archive = cache / "efficientad_pretrained_weights.zip"
    name = f"pretrained_teacher_{settings.get('model_size', 'small')}.pth"
    target = cache / name
    manifest_path = target.with_suffix(".integrity.json")
    if Path(filesystem_path(target)).is_file() and Path(filesystem_path(manifest_path)).is_file():
        try:
            with open(filesystem_path(manifest_path), "r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            if (manifest.get("archive_sha256") == TEACHER_SHA256
                    and manifest.get("source") == TEACHER_URL
                    and sha256_file(target) == manifest.get("sha256")):
                return target, manifest
        except (OSError, ValueError):
            pass
    if not Path(filesystem_path(archive)).is_file() or sha256_file(archive) != TEACHER_SHA256:
        if not settings.get("allow_download", True):
            raise ModelSetupError(
                f"Verified EfficientAD teacher unavailable. Enable allow_download or cache "
                f"the official archive at {archive}. Expected SHA256: {TEACHER_SHA256}"
            )
        verified_download(TEACHER_URL, archive, TEACHER_SHA256, progress=progress)
    with open(filesystem_path(archive), "rb") as archive_file, zipfile.ZipFile(archive_file) as source:
        matches = [entry for entry in source.infolist() if Path(entry.filename).name == name and not entry.is_dir()]
        if len(matches) != 1:
            raise ModelSetupError(f"Official EfficientAD archive does not contain exactly one {name}.")
        if matches[0].file_size > 100 * 1024 * 1024:
            raise ModelSetupError("Unexpected teacher checkpoint size.")
        # Copy one known basename rather than extracting archive paths.
        with source.open(matches[0]) as src, open(filesystem_path(target), "wb") as dst:
            shutil.copyfileobj(src, dst)
    manifest = {"source": TEACHER_URL, "archive_sha256": TEACHER_SHA256,
                "sha256": sha256_file(target), "authenticity": "published_anomalib_2.2.0_archive_sha256"}
    with open(filesystem_path(manifest_path), "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    return target, manifest


def _image_paths(root: Path) -> list[Path]:
    filesystem_root = Path(filesystem_path(root))
    if not filesystem_root.is_dir():
        return []
    return sorted(root / path.relative_to(filesystem_root)
                  for path in filesystem_root.rglob("*")
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def natural_image_paths(settings: dict, *, progress: Callable[[str], None] | None = None) -> list[Path]:
    """Find genuine natural images for EfficientAD's required penalty term.

    Use the configured natural-image folder when populated. Otherwise download
    and verify official Imagenette automatically, including installations with
    the obsolete download_imagenette setting disabled. Aluminium GOOD images
    are never used as a replacement for this required penalty dataset.
    """
    root = Path(settings.get("imagenet_dir") or "artifacts/pretrained/imagenette2/train").expanduser()
    emit = progress or (lambda _: None)
    paths = _image_paths(root)
    if paths:
        emit(f"Using {len(paths)} cached natural photographs from {root}")
        return paths
    emit(f"Natural-image dataset missing at {root}; preparing official Imagenette automatically (~1.5 GB)")
    cache = Path(settings.get("teacher_cache") or "artifacts/pretrained/efficientad").expanduser().parent
    archive = verified_download(IMAGENETTE_URL, cache / "imagenette2.tgz", IMAGENETTE_SHA256, progress=progress)
    emit("Extracting verified Imagenette training photographs")
    count = 0
    with open(filesystem_path(archive), "rb") as archive_file, tarfile.open(fileobj=archive_file, mode="r:gz") as source:
        for member in source:
            parts = Path(member.name).parts
            # Only regular training images inside the expected dataset directory.
            if (not member.isfile() or len(parts) < 4 or parts[0:2] != ("imagenette2", "train")
                    or ".." in parts or Path(member.name).suffix.lower() not in IMAGE_SUFFIXES):
                continue
            output = root.joinpath(*parts[2:])
            Path(filesystem_path(output.parent)).mkdir(parents=True, exist_ok=True)
            stream = source.extractfile(member)
            if stream is not None:
                with stream, open(filesystem_path(output), "wb") as target:
                    shutil.copyfileobj(stream, target)
                count += 1
                if count % 500 == 0:
                    emit(f"Extracting Imagenette: {count} training photographs")
    paths = _image_paths(root)
    if not paths:
        raise ModelSetupError("The verified ImageNette archive contains no usable training photographs.")
    emit(f"Imagenette ready: {len(paths)} training photographs")
    return paths
