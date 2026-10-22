"""View-aware image provenance and physical-part grouping.

Original bytes and valid provenance remain immutable. Damaged sidecars can be
restored from a verified mirror or explicitly repaired after preserving a backup.
Rejecting a sample adds a tombstone.
A folder name is never used as evidence of an independent physical component.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import os
import random
import re
import threading
import uuid

import cv2
import numpy as np
from src.utils.filesystem import display_path, filesystem_path

VIEWS = ("inner_face", "ribbed_face", "side_left", "side_right")
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_WINDOWS = os.name == "nt"


class DatasetIntegrityError(ValueError):
    """An image's provenance or label is missing, corrupt, or contradictory."""


@dataclass(frozen=True)
class Sample:
    id: str
    path: Path
    label: int
    view: str
    part_id: str
    sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def session_id(self) -> str:
        return str(self.metadata.get("session_id", ""))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "path": display_path(self.path), "label": self.label,
                "view": self.view, "part_id": self.part_id, "sha256": self.sha256,
                "metadata": self.metadata}


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decode(data: bytes, description: str = "image") -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise DatasetIntegrityError(f"Unreadable image: {description}")
    return image


def _phash(image: np.ndarray) -> str:
    small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
    low = cv2.dct(small)[:8, :8].ravel()
    bits = low > np.median(low[1:])
    return f"{int(''.join('1' if x else '0' for x in bits), 2):016x}"


def _validate_identity(view: str, label: str | int, part_id: str) -> int:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", view):
        raise ValueError("View must be a simple name such as inner_face or ribbed_face")
    if label not in ("GOOD", "NG", 0, 1):
        raise ValueError("Label must be GOOD or NG")
    if not isinstance(part_id, str) or not part_id.strip():
        raise ValueError("A real physical part ID is required; captures are not independent parts")
    return 0 if label in ("GOOD", 0) else 1


class _MetadataRepairNeeded(DatasetIntegrityError):
    """Missing identity evidence requires an explicit operator assignment."""


def _publish_new(staged: Path, target: Path) -> None:
    """Publish a complete same-directory file without replacing another file.

    Windows rename refuses an existing destination and works without hard-link
    support on FAT/exFAT and network shares. POSIX rename would replace a file,
    so use an exclusive hard link there. Callers clean up remaining staging links.
    """
    staged, target = filesystem_path(staged), filesystem_path(target)
    if _WINDOWS:
        os.rename(staged, target)
    else:
        os.link(staged, target)


def _atomic_bytes(path: Path, payload: bytes, *, replace: bool = False) -> None:
    """Publish only a closed, fsynced file; never expose a partial final JSON."""
    path = filesystem_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".metadata-{uuid.uuid4().hex}.tmp")
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
            created = False
        else:
            _publish_new(temporary, path)
            temporary.unlink(missing_ok=True)
            created = False
        _sync_directory(path.parent)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def _sync_directory(directory: Path) -> None:
    # Directory fsync is supported on POSIX; Windows cannot open a directory this way.
    if os.name != "nt":
        handle = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")


def _slug(value: str, maximum: int, fallback: str) -> str:
    # Dots are excluded too: Windows interprets CON.anything as a device name.
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")[:maximum] or fallback


def _canonical_path(path: str | Path) -> Path:
    """Resolve links using an OS-usable path, then restore Windows long-path form."""
    return filesystem_path(filesystem_path(path).resolve())


class DatasetStore:
    _root_locks: dict[str, Any] = {}
    _root_locks_guard = threading.Lock()

    def __init__(self, root: str | Path):
        self.root = _canonical_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        with self._root_locks_guard:
            self._lock = self._root_locks.setdefault(os.path.normcase(str(self.root)), threading.RLock())

    def _internal(self, path: Path) -> bool:
        path = filesystem_path(path)
        return not path.is_symlink() and _canonical_path(path).is_relative_to(self.root)

    @staticmethod
    def _sidecar(path: Path) -> Path:
        path = filesystem_path(path)
        return path.with_suffix(path.suffix + ".json")

    def _relative(self, path: Path) -> Path:
        return _canonical_path(path).relative_to(self.root)

    def _mirror(self, path: Path) -> Path:
        relative = self._relative(path).as_posix()
        key = hashlib.sha256(relative.encode("utf-8")).hexdigest()
        mirror = self.root / ".metadata" / (key + ".json")
        if not _canonical_path(mirror).is_relative_to(self.root) or mirror.is_symlink():
            raise DatasetIntegrityError("Metadata recovery directory/file uses an unsafe path")
        return mirror

    def _paths(self):
        for path in sorted(self.root.glob("*/*/*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS and path.parent.name in ("GOOD", "NG") and not path.parent.parent.name.startswith("."):
                if not self._internal(path):
                    raise DatasetIntegrityError(f"Dataset image uses an unsafe external/symlink path: {display_path(path)}")
                yield path, path.parent.parent.name, path.parent.name
        for folder in ("GOOD", "NG"):
            for path in sorted((self.root / folder).glob("*")):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    if not self._internal(path):
                        raise DatasetIntegrityError(f"Dataset image uses an unsafe external/symlink path: {display_path(path)}")
                    yield path, None, folder

    def _validate_details(self, details: Any, path: Path, view: str | None,
                          folder: str, digest: str) -> dict[str, Any]:
        if not isinstance(details, dict):
            raise _MetadataRepairNeeded(f"Invalid provenance metadata: {display_path(self._sidecar(path))}: expected a JSON object")
        # Any available contradictory evidence remains fatal, even in an incomplete file.
        claimed_hash = details.get("sha256")
        if isinstance(claimed_hash, str) and re.fullmatch(r"[a-f0-9]{64}", claimed_hash) and claimed_hash != digest:
            raise DatasetIntegrityError(f"Image bytes differ from immutable sidecar: {display_path(path)}")
        if view is not None and details.get("view") is not None and details["view"] != view:
            raise DatasetIntegrityError(f"Image view differs from immutable sidecar: {display_path(path)}")
        label = details.get("label")
        if label in (0, 1, "GOOD", "NG") and (0 if label in (0, "GOOD") else 1) != (0 if folder == "GOOD" else 1):
            raise DatasetIntegrityError(f"Image label differs from immutable sidecar: {display_path(path)}")
        if details.get("schema_version", 1) != 1:
            raise DatasetIntegrityError(f"Unsupported provenance schema: {display_path(path)}")
        required = ("schema_version", "id", "view", "label", "part_id", "sha256")
        if any(key not in details for key in required):
            raise _MetadataRepairNeeded(f"Invalid provenance metadata: {display_path(self._sidecar(path))}: missing required identity fields; repair with explicit physical part ID")
        try:
            numeric = _validate_identity(details["view"], details["label"], details["part_id"])
        except (TypeError, ValueError) as exc:
            raise _MetadataRepairNeeded(f"Invalid provenance metadata: {display_path(self._sidecar(path))}: {exc}") from exc
        if numeric != details["label"] or not isinstance(details["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", details["id"]):
            raise _MetadataRepairNeeded(f"Invalid provenance metadata: {display_path(self._sidecar(path))}: invalid sample ID or numeric label")
        if not isinstance(claimed_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", claimed_hash):
            raise _MetadataRepairNeeded(f"Invalid provenance metadata: {display_path(self._sidecar(path))}: invalid source hash")
        if view is None:
            raise _MetadataRepairNeeded(f"Legacy image needs explicit view registration: {display_path(path)}")
        return details

    def _archive_sidecar(self, path: Path, original: bytes, digest: str,
                         selection: dict[str, Any] | None = None) -> Path:
        backup = self.root / ".metadata" / "repairs" / (uuid.uuid4().hex + ".sidecar.bin")
        if not _canonical_path(backup).is_relative_to(self.root):
            raise DatasetIntegrityError("Metadata repair backup directory uses an unsafe path")
        _atomic_bytes(backup, original)
        _atomic_bytes(backup.with_suffix(".json"), _json_bytes({
            "schema_version": 1, "original_image": display_path(path), "image_sha256": digest,
            "original_sidecar": display_path(self._sidecar(path)), "sidecar_sha256": _hash(original),
            "saved_at": datetime.now(timezone.utc).isoformat(), "selection": selection}))
        return backup

    def _details(self, path: Path, view: str | None, folder: str) -> tuple[dict[str, Any], bytes]:
        path = filesystem_path(path)
        data = path.read_bytes()
        digest = _hash(data)
        sidecar = self._sidecar(path)
        if sidecar.is_symlink():
            raise DatasetIntegrityError(f"Metadata sidecar uses an unsafe symbolic link: {display_path(sidecar)}")
        original = sidecar.read_bytes() if sidecar.exists() else None
        error = None
        try:
            if original is None:
                raise _MetadataRepairNeeded(f"Missing metadata/physical part ID: {display_path(path)}. Import using Dataset Manager with explicit IDs.")
            try:
                parsed = json.loads(original.decode("utf-8-sig"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise _MetadataRepairNeeded(f"Invalid provenance metadata: {display_path(sidecar)}: {exc}; repair with explicit physical part ID") from exc
            details = self._validate_details(parsed, path, view, folder, digest)
        except _MetadataRepairNeeded as exc:
            error = exc
            details = None
        mirror = self._mirror(path)
        if mirror.exists():
            try:
                wrapper = json.loads(mirror.read_text(encoding="utf-8-sig"))
                recovered = wrapper["metadata"]
                if (wrapper["schema_version"] != 1 or wrapper["relative_path"] != self._relative(path).as_posix()
                        or wrapper["image_sha256"] != digest or wrapper["metadata_sha256"] != _hash(_json_bytes(recovered))):
                    raise ValueError("recovery mirror path, image hash or metadata checksum mismatch")
                recovered = self._validate_details(recovered, path, view, folder, digest)
            except (KeyError, TypeError, ValueError, UnicodeError) as exc:
                raise DatasetIntegrityError(f"Untrusted/corrupt metadata recovery mirror: {display_path(mirror)}: {exc}") from exc
            if details is None:
                if original is not None:
                    self._archive_sidecar(path, original, digest)
                _atomic_bytes(sidecar, _json_bytes(recovered), replace=original is not None)
                details = recovered
            elif _json_bytes(details) != _json_bytes(recovered):
                raise DatasetIntegrityError(f"Provenance identity differs from immutable metadata recovery mirror: {display_path(path)}")
        if details is None:
            raise error
        _decode(data, display_path(path))
        return details, data

    def import_image(self, path: str | Path, view: str, label: str | int,
                     part_id: str, session_id: str = "", *,
                     metadata: dict[str, Any] | None = None) -> Sample:
        source = filesystem_path(path)
        if source.is_symlink():
            raise DatasetIntegrityError("Select an original image, not a symbolic link")
        source = _canonical_path(source)
        if source.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source.suffix}")
        _validate_identity(view, label, part_id)
        with self._lock:
            if self._internal(source) and self._sidecar(source).exists():
                candidate_view = source.parent.parent.name if source.parent.name in ("GOOD", "NG") and source.parent.parent != self.root else None
                folder = source.parent.name if source.parent.name in ("GOOD", "NG") else ("GOOD" if label in (0, "GOOD") else "NG")
                try:
                    self._details(source, candidate_view, folder)
                except _MetadataRepairNeeded:
                    return self.repair_image(source, view, label, part_id, session_id, metadata=metadata)
            return self._store(source.read_bytes(), source.suffix.lower(), view, label,
                               part_id, session_id, display_path(source), metadata, source_path=source)

    def repair_image(self, path: str | Path, view: str, label: str | int,
                     part_id: str, session_id: str = "", *,
                     metadata: dict[str, Any] | None = None) -> Sample:
        source = filesystem_path(path)
        if source.is_symlink():
            raise DatasetIntegrityError("Select an original image, not a symbolic link")
        source = _canonical_path(source)
        numeric = _validate_identity(view, label, part_id)
        if not self._internal(source) or source.suffix.lower() not in EXTENSIONS:
            raise DatasetIntegrityError("Metadata repair requires an original image inside this dataset")
        with self._lock:
            sidecar = self._sidecar(source)
            if not sidecar.exists():
                return self.import_image(source, view, label, part_id, session_id, metadata=metadata)
            candidate_view = source.parent.parent.name if source.parent.name in ("GOOD", "NG") and source.parent.parent != self.root else None
            folder = source.parent.name if source.parent.name in ("GOOD", "NG") else ("GOOD" if numeric == 0 else "NG")
            try:
                self._details(source, candidate_view, folder)
            except _MetadataRepairNeeded:
                pass
            else:
                # Intact or safely recovered records cannot be relabelled by repair.
                return self._store(source.read_bytes(), source.suffix.lower(), view, label,
                                   part_id, session_id, display_path(source), metadata, source_path=source)
            data = source.read_bytes()
            _decode(data, display_path(source))
            broken = sidecar.read_bytes()
            selection = {"view": view, "label": numeric, "part_id": part_id.strip(), "session_id": str(session_id)}
            self._check_partial_identity(broken, selection)
            backup = self._archive_sidecar(source, broken, _hash(data), selection)
            extra = {**(metadata or {}), "repaired_from": display_path(source),
                     "metadata_repair_backup": str(self._relative(backup))}
            return self._store(data, source.suffix.lower(), view, label, part_id, session_id,
                               display_path(source), extra, source_path=source,
                               repair_sidecar_sha256=_hash(broken))

    @staticmethod
    def _check_partial_identity(payload: bytes, selection: dict[str, Any]) -> None:
        """Incomplete JSON can still contain evidence that must not be relabelled."""
        try:
            details = json.loads(payload.decode("utf-8-sig"))
        except (UnicodeError, json.JSONDecodeError):
            return
        if not isinstance(details, dict):
            return
        for field in ("part_id", "view", "label"):
            value = details.get(field)
            if field == "label" and value in ("GOOD", "NG"):
                value = 0 if value == "GOOD" else 1
            if field == "part_id" and isinstance(value, str):
                value = value.strip()
            if value is not None and value != selection[field]:
                raise DatasetIntegrityError(f"Existing metadata {field} conflicts with the selected identity; explicit repair cannot relabel known evidence")

    def capture_image(self, image: np.ndarray, view: str, label: str | int,
                      part_id: str, session_id: str = "", *,
                      metadata: dict[str, Any] | None = None) -> Sample:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Capture must be a native-resolution uint8 BGR image")
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Could not encode captured image")
        return self._store(encoded.tobytes(), ".png", view, label, part_id,
                           session_id, "camera", metadata)

    def _finish_source(self, source: Path | None, sample: Sample, digest: str,
                       repair_sidecar_sha256: str | None) -> None:
        if source is None:
            return
        source = filesystem_path(source)
        if not self._internal(source) or _canonical_path(source) == _canonical_path(sample.path):
            return
        sidecar = self._sidecar(source)
        if sidecar.exists() and repair_sidecar_sha256 is None:
            return  # A managed source record is never removed during ordinary import.
        try:
            if not source.is_file() or _hash(source.read_bytes()) != digest:
                raise DatasetIntegrityError("Selected source changed during registration")
            if repair_sidecar_sha256 is not None:
                if not sidecar.exists() or _hash(sidecar.read_bytes()) != repair_sidecar_sha256:
                    raise DatasetIntegrityError("Selected source metadata changed during repair")
            elif sidecar.exists():
                raise DatasetIntegrityError("Selected source gained metadata during registration")
            source.unlink()
            if repair_sidecar_sha256 is not None:
                sidecar.unlink()
        except (OSError, DatasetIntegrityError) as exc:
            raise DatasetIntegrityError(
                f"Verified managed copy retained at {display_path(sample.path)}, but the selected old source could not be cleaned up: {display_path(source)}: {exc}. Close the file and retry registration; neither copy is overwritten.") from exc

    def _store(self, data: bytes, suffix: str, view: str, label: str | int,
               part_id: str, session_id: str, source: str,
               extra: dict[str, Any] | None, *, source_path: Path | None = None,
               repair_sidecar_sha256: str | None = None) -> Sample:
        with self._lock:
            numeric_label = _validate_identity(view, label, part_id)
            part_id = part_id.strip()
            image = _decode(data, source)
            digest = _hash(data)
            for tombstone in (self.root / ".rejected").glob("*.json"):
                try:
                    rejection = json.loads(tombstone.read_text(encoding="utf-8-sig"))
                    if rejection["sha256"] == digest:
                        raise DatasetIntegrityError("This exact duplicate image was rejected; importing or repairing it cannot silently restore it")
                except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
                    raise DatasetIntegrityError(f"Invalid sample rejection record: {display_path(tombstone)}") from exc
            selection = {"part_id": part_id, "view": view, "label": numeric_label}
            for candidate, candidate_view, folder in self._paths():
                sidecar = self._sidecar(candidate)
                if sidecar.exists() and _hash(candidate.read_bytes()) == digest:
                    # A malformed/incomplete record may not hide contradictory
                    # identity evidence merely because unrelated imports can proceed.
                    self._check_partial_identity(sidecar.read_bytes(), selection)
            for existing in self._scan(include_rejected=True, strict=False):
                if existing.sha256 == digest:
                    if (existing.part_id, existing.view, existing.label) != (part_id, view, numeric_label):
                        raise DatasetIntegrityError(f"Exact duplicate claims conflicting identity/label: {existing.id} ({existing.part_id}, {existing.view}); original retained")
                    if (self.root / ".rejected" / f"{existing.id}.json").exists():
                        raise DatasetIntegrityError("This exact image was rejected; importing it cannot silently restore it")
                    self._finish_source(source_path, existing, digest, repair_sidecar_sha256)
                    return existing
            sample_id = uuid.uuid4().hex
            directory = self.root / view / ("GOOD" if numeric_label == 0 else "NG")
            if not _canonical_path(directory).is_relative_to(self.root):
                raise DatasetIntegrityError("Selected dataset view directory uses an unsafe path")
            directory.mkdir(parents=True, exist_ok=True)
            name = "_".join((_slug(part_id, 24, "part"), _slug(view, 20, "view"),
                             "GOOD" if numeric_label == 0 else "NG",
                             _slug(str(session_id), 16, "no-session"), sample_id[:12]))
            target = directory / (name + suffix)
            details = dict(extra or {})
            details.update({"schema_version": 1, "id": sample_id, "view": view,
                            "label": numeric_label, "part_id": part_id,
                            "session_id": str(session_id), "sha256": digest,
                            "perceptual_hash": _phash(image), "source": source,
                            "original_filename": source_path.name if source_path is not None else None,
                            "imported_at": datetime.now(timezone.utc).isoformat(),
                            "captured_at": datetime.now(timezone.utc).isoformat() if source == "camera" else None,
                            "width": int(image.shape[1]), "height": int(image.shape[0])})
            sidecar = self._sidecar(target)
            mirror = self._mirror(target)
            created_image = created_sidecar = created_mirror = False
            staged = directory / (".image-" + uuid.uuid4().hex + ".tmp")
            created_staged = False
            try:
                if target.exists():
                    raise FileExistsError(f"Refusing to overwrite existing dataset image: {target}")
                with staged.open("xb") as stream:
                    created_staged = True
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                wrapper = {"schema_version": 1, "relative_path": self._relative(target).as_posix(),
                           "image_sha256": digest, "metadata_sha256": _hash(_json_bytes(details)),
                           "metadata": details}
                _atomic_bytes(mirror, _json_bytes(wrapper))
                created_mirror = True
                _atomic_bytes(sidecar, _json_bytes(details))
                created_sidecar = True
                # Publish the complete image last. A scanner cannot observe a
                # new image until both durable metadata copies already exist.
                _publish_new(staged, target)
                created_image = True
                staged.unlink(missing_ok=True)
                created_staged = False
                _sync_directory(directory)
                verified, original = self._details(target, view, "GOOD" if numeric_label == 0 else "NG")
                if _hash(original) != digest or verified != details:
                    raise DatasetIntegrityError("Newly written dataset record failed verification")
            except Exception:
                # Exclusive-open failures do not grant ownership of pre-existing files.
                if created_sidecar:
                    sidecar.unlink(missing_ok=True)
                if created_mirror:
                    mirror.unlink(missing_ok=True)
                if created_image:
                    target.unlink(missing_ok=True)
                raise
            finally:
                if created_staged:
                    staged.unlink(missing_ok=True)
            sample = Sample(sample_id, target, numeric_label, view, part_id, digest, details)
            self._finish_source(source_path, sample, digest, repair_sidecar_sha256)
            return sample

    def _scan(self, include_rejected: bool = False, view: str | None = None,
              strict: bool = True) -> list[Sample]:
        with self._lock:
            samples = []
            seen_ids = set()
            hashes = {}
            for path, candidate_view, folder in self._paths():
                try:
                    details, data = self._details(path, candidate_view, folder)
                except _MetadataRepairNeeded:
                    if strict:
                        raise
                    continue
                sample_id = details["id"]
                if sample_id in seen_ids:
                    raise DatasetIntegrityError(f"Repeated sample ID: {sample_id}")
                seen_ids.add(sample_id)
                sample = Sample(sample_id, path, details["label"], candidate_view,
                                details["part_id"].strip(), _hash(data), details)
                prior = hashes.get(sample.sha256)
                if prior and (prior.part_id, prior.label, prior.view) != (sample.part_id, sample.label, sample.view):
                    raise DatasetIntegrityError(f"Duplicate bytes claim incompatible labels/physical parts/views: {display_path(prior.path)}, {display_path(path)}")
                hashes[sample.sha256] = sample
                if (include_rejected or not (self.root / ".rejected" / f"{sample_id}.json").exists()) and (view is None or candidate_view == view):
                    samples.append(sample)
            return samples

    def inventory(self, view: str | None = None, strict: bool = True) -> list[Sample]:
        return self._scan(view=view, strict=strict)

    def unmanaged_images(self, view: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            pending = []
            for path, candidate_view, folder in self._paths():
                try:
                    self._details(path, candidate_view, folder)
                except _MetadataRepairNeeded as exc:
                    if view is None or candidate_view == view:
                        pending.append({"path": display_path(path), "view": candidate_view, "label": folder,
                                        "status": "repair_required" if self._sidecar(path).exists() else "unassigned",
                                        "reason": str(exc)})
            return pending

    def remove(self, sample_id: str, reason: str = "operator rejected sample") -> None:
        with self._lock:
            sample = next((s for s in self._scan(include_rejected=True, strict=False) if s.id == sample_id), None)
            if sample is None:
                raise KeyError(f"Unknown dataset sample: {sample_id}")
            target = self.root / ".rejected" / f"{sample.id}.json"
            if not target.exists():
                _atomic_bytes(target, _json_bytes({"id": sample.id, "sha256": sample.sha256, "reason": reason,
                                                 "rejected_at": datetime.now(timezone.utc).isoformat()}))

    def stats(self, *, group_sessions: bool = False) -> dict[str, Any]:
        with self._lock:
            all_samples = self._scan(include_rejected=True, strict=False)
            active = self.inventory(strict=False)
            views: dict[str, dict[str, int]] = {}
            hashes: dict[str, list[str]] = {}
            near: list[dict[str, Any]] = []
            for index, sample in enumerate(active):
                bucket = views.setdefault(sample.view, {"GOOD": 0, "NG": 0, "parts": 0})
                bucket["GOOD" if sample.label == 0 else "NG"] += 1
                hashes.setdefault(sample.sha256, []).append(sample.id)
                left = sample.metadata.get("perceptual_hash")
                if left:
                    for other in active[:index]:
                        right = other.metadata.get("perceptual_hash")
                        if right and sample.sha256 != other.sha256:
                            distance = (int(left, 16) ^ int(right, 16)).bit_count()
                            if distance <= 6:
                                near.append({"samples": [other.id, sample.id], "hamming_distance": distance,
                                             "warning": "Visually similar captures are not proof of independent parts"})
            for candidate_view, bucket in views.items():
                bucket["parts"] = len({s.part_id for s in active if s.view == candidate_view})
            readiness = training_readiness(active, group_sessions=group_sessions)
            pending = self.unmanaged_images()
            return {"images": len(active), "parts": readiness["parts"],
                    "good_images": readiness["good_images"], "ng_images": readiness["ng_images"],
                    "good_parts": readiness["good_parts"], "ng_parts": readiness["ng_parts"],
                    "training_readiness": readiness,
                    "rejected": len(all_samples) - len(active), "views": views,
                    "duplicates": {k: v for k, v in hashes.items() if len(v) > 1},
                    "near_duplicate_warnings": near, "unmanaged_count": len(pending),
                    "unmanaged_images": pending, "ng_sensitivity_validated": False}


@dataclass
class DatasetSplit:
    train: list[Sample]
    validation: list[Sample]
    test: list[Sample]
    seed: int = 42

    def manifest(self) -> dict[str, Any]:
        return {"schema_version": 1, "seed": self.seed,
                **{name: [s.to_dict() for s in getattr(self, name)]
                   for name in ("train", "validation", "test")}}


def _training_groups(samples: list[Sample], group_sessions: bool) -> dict[str, list[Sample]]:
    """Validate provenance claims and group parts exactly as the split does."""
    parents = {s.part_id: s.part_id for s in samples}
    def find(value: str) -> str:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value
    hashes: dict[str, Sample] = {}
    sessions: dict[str, str] = {}
    for sample in samples:
        _validate_identity(sample.view, sample.label, sample.part_id)
        prior = hashes.get(sample.sha256)
        if prior and (prior.part_id != sample.part_id or prior.label != sample.label or prior.view != sample.view):
            raise DatasetIntegrityError("Duplicate image bytes claim conflicting physical parts, views or labels")
        hashes[sample.sha256] = sample
        if group_sessions and sample.session_id:
            previous = sessions.setdefault(sample.session_id, sample.part_id)
            parents[find(sample.part_id)] = find(previous)
    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        groups.setdefault(find(sample.part_id), []).append(sample)
    return groups


def _example_ids(values: Iterable[str], limit: int = 2) -> str:
    """Keep messages usable when identities are numerous or unusually long."""
    values = sorted(set(values))
    examples = [repr(value if len(value) <= 32 else value[:29] + "...") for value in values[:limit]]
    if len(values) > limit:
        examples.append(f"and {len(values) - limit} more")
    return ", ".join(examples)


def _readiness_report(samples: list[Sample], groups: dict[str, list[Sample]],
                      group_sessions: bool) -> dict[str, Any]:
    by_part: dict[str, list[Sample]] = {}
    for sample in samples:
        by_part.setdefault(sample.part_id, []).append(sample)
    details = []
    for part_id, captures in sorted(by_part.items()):
        good_images = sum(s.label == 0 for s in captures)
        ng_images = len(captures) - good_images
        details.append({"part_id": part_id, "images": len(captures),
                        "good_images": good_images, "ng_images": ng_images,
                        "label": "NG" if ng_images else "GOOD",
                        "mixed_labels": bool(good_images and ng_images),
                        "views": sorted({s.view for s in captures}),
                        "session_ids": sorted({s.session_id for s in captures if s.session_id})})
    good_parts = sum(part["label"] == "GOOD" for part in details)
    ng_parts = len(details) - good_parts
    mixed_parts = sum(part["mixed_labels"] for part in details)
    good_images = sum(s.label == 0 for s in samples)
    good_groups = sum(not any(s.label for s in captures) for captures in groups.values())
    ng_groups = len(groups) - good_groups
    can_split = good_groups >= 3
    messages = [f"{len(samples)} images ({good_images} GOOD, {len(samples) - good_images} NG); "
                f"{len(details)} physical housing(s): {good_parts} GOOD, {ng_parts} NG "
                f"({mixed_parts} mixed-label). Independent GOOD groups: {good_groups}/3."]
    if len(details) == 1:
        messages.append(f"All images share physical part ID {_example_ids(by_part)}. "
                        "Repeated photos still count as one housing.")
    if ng_parts:
        examples = _example_ids(part["part_id"] for part in details if part["ng_images"])
        messages.append("Any NG-labelled capture excludes all views of its housing from normal "
                        f"training, even with other GOOD photos (IDs: {examples}).")
    if group_sessions:
        messages.append(f"Session grouping: {len(details)} physical housing(s) form "
                        f"{len(groups)} independent group(s).")
        sessions: dict[str, set[str]] = {}
        for sample in samples:
            if sample.session_id:
                sessions.setdefault(sample.session_id, set()).add(sample.part_id)
        shared = [session for session, part_ids in sessions.items() if len(part_ids) > 1]
        if shared:
            messages.append(f"Shared capture sessions ({_example_ids(shared)}) keep parts together; "
                            "NG excludes its whole session group.")
    if can_split:
        messages.append("Group minimum met. Selected-face coverage and image quality still need validation.")
    else:
        messages.append("At least 3 independent GOOD parts/groups are needed for separate "
                        "train/validation/test sets.")
        if samples:
            messages.append("Review IDs and labels in Dataset Manager; keep originals before correcting "
                            "mistaken imports. Capture more actual GOOD housings; never invent IDs for repeats.")
        else:
            messages.append("Import captures of actual GOOD housings in Dataset Manager, using the same "
                            "physical part ID for every capture of each housing.")
    return {"images": len(samples), "good_images": good_images, "ng_images": len(samples) - good_images,
            "parts": len(details), "good_parts": good_parts, "ng_parts": ng_parts,
            "mixed_label_parts": mixed_parts, "part_details": details,
            "eligible_good_groups": good_groups, "eligible_ng_groups": ng_groups,
            "group_sessions": bool(group_sessions), "minimum_good_groups": 3,
            "can_split": can_split, "message": " ".join(messages)}


def training_readiness(samples: Iterable[Sample], *, group_sessions: bool = False) -> dict[str, Any]:
    """Explain the global independent-GOOD-group requirement without changing data.

    ``can_split`` covers only the minimum global group count. The training service
    also checks selected-face coverage, image quality and backend prerequisites.
    Duplicate/conflicting identity claims fail exactly as they do during splitting.
    """
    samples = list(samples)
    return _readiness_report(samples, _training_groups(samples, group_sessions), group_sessions)


def split_by_part(samples: Iterable[Sample], seed: int = 42,
                  ratios: tuple[float, float, float] = (.7, .15, .15), *,
                  group_sessions: bool = False) -> DatasetSplit:
    """Split globally across views; NG physical parts never enter normal training.

    Session grouping is opt-in because a single acquisition day may include the
    entire initial dataset. When enabled, connected session/part groups stay whole.
    """
    samples = list(samples)
    if len(ratios) != 3 or any(not np.isfinite(x) or x <= 0 for x in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError("Three positive split ratios summing to one are required")
    groups = _training_groups(samples, group_sessions)
    good = sorted(key for key, values in groups.items() if not any(s.label for s in values))
    ng = sorted(key for key, values in groups.items() if any(s.label for s in values))
    if len(good) < 3:
        raise ValueError(_readiness_report(samples, groups, group_sessions)["message"])
    rng = random.Random(seed)
    rng.shuffle(good)
    rng.shuffle(ng)
    n = len(good)
    n_val = max(1, round(n * ratios[1]))
    n_test = max(1, round(n * ratios[2]))
    while n_val + n_test >= n:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            raise ValueError("Not enough independent GOOD groups for these partitions")
    keys = [good[:n-n_val-n_test], good[n-n_val-n_test:n-n_test], good[n-n_test:]]
    # NG is reserved for threshold trade-offs and untouched test sensitivity.
    n_ng_val = max(1, round(len(ng) * ratios[1] / (ratios[1] + ratios[2]))) if len(ng) > 1 else 0
    keys[1].extend(ng[:n_ng_val])
    keys[2].extend(ng[n_ng_val:])
    partitions = [[s for key in group_keys for s in groups[key]] for group_keys in keys]
    return DatasetSplit(*partitions, seed=seed)
