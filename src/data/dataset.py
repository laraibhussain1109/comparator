"""View-aware image provenance and physical-part grouping.

Source bytes and sidecars are write-once; rejecting a sample adds a tombstone.
A folder name is never used as evidence of an independent physical component.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import random
import re
import uuid

import cv2
import numpy as np

VIEWS = ("inner_face", "ribbed_face", "side_left", "side_right")
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


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
        return {"id": self.id, "path": str(self.path), "label": self.label,
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


class DatasetStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def import_image(self, path: str | Path, view: str, label: str | int,
                     part_id: str, session_id: str = "", *,
                     metadata: dict[str, Any] | None = None) -> Sample:
        source = Path(path)
        if source.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source.suffix}")
        return self._store(source.read_bytes(), source.suffix.lower(), view, label,
                           part_id, session_id, str(source.resolve()), metadata)

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

    def _store(self, data: bytes, suffix: str, view: str, label: str | int,
               part_id: str, session_id: str, source: str,
               extra: dict[str, Any] | None) -> Sample:
        numeric_label = _validate_identity(view, label, part_id)
        part_id = part_id.strip()
        image = _decode(data, source)
        digest = _hash(data)
        # Scan rejected samples too: a rejected original cannot be relabelled through reimport.
        for existing in self._scan(include_rejected=True):
            if existing.sha256 == digest:
                raise DatasetIntegrityError(f"Exact duplicate of {existing.id} ({existing.part_id}, {existing.view}); original retained")
        sample_id = uuid.uuid4().hex
        directory = self.root / view / ("GOOD" if numeric_label == 0 else "NG")
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / (sample_id + suffix)
        details = dict(extra or {})
        # Reserved provenance fields cannot be overridden by imported user metadata.
        details.update({"schema_version": 1, "id": sample_id, "view": view,
                        "label": numeric_label, "part_id": part_id,
                        "session_id": str(session_id), "sha256": digest,
                        "perceptual_hash": _phash(image), "source": source,
                        "imported_at": datetime.now(timezone.utc).isoformat(),
                        "captured_at": datetime.now(timezone.utc).isoformat() if source == "camera" else None,
                        "width": int(image.shape[1]), "height": int(image.shape[0])})
        payload = json.dumps(details, indent=2, sort_keys=True, allow_nan=False)
        sidecar = target.with_suffix(target.suffix + ".json")
        try:
            with target.open("xb") as stream:
                stream.write(data)
            with sidecar.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        except Exception:
            # Only this fresh UUID's files may be cleaned up after an incomplete import.
            target.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            raise
        return Sample(sample_id, target, numeric_label, view, part_id, digest, details)

    def _scan(self, include_rejected: bool = False, view: str | None = None) -> list[Sample]:
        samples: list[Sample] = []
        seen_ids: set[str] = set()
        hashes: dict[str, Sample] = {}
        for path in sorted(self.root.glob("*/*/*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            candidate_view, folder = path.parent.parent.name, path.parent.name
            if folder not in ("GOOD", "NG"):
                continue
            sidecar = path.with_suffix(path.suffix + ".json")
            if not sidecar.is_file():
                raise DatasetIntegrityError(f"Missing metadata/physical part ID: {path}. Import using Dataset Manager; never infer part IDs from image names.")
            try:
                details = json.loads(sidecar.read_text(encoding="utf-8"))
                sample_id = details["id"]
                label = _validate_identity(candidate_view, details["label"], details["part_id"])
                original = path.read_bytes()
                digest = _hash(original)
                if label != (0 if folder == "GOOD" else 1) or details["view"] != candidate_view or details["sha256"] != digest:
                    raise DatasetIntegrityError(f"Image bytes, view or label differ from immutable sidecar: {path}")
                if not isinstance(sample_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
                    raise DatasetIntegrityError(f"Invalid sample ID: {path}")
                if sample_id in seen_ids:
                    raise DatasetIntegrityError(f"Repeated sample ID: {sample_id}")
                seen_ids.add(sample_id)
                _decode(original, str(path))
                sample = Sample(sample_id, path, label, candidate_view, details["part_id"].strip(), digest, details)
                prior = hashes.get(digest)
                if prior and (prior.part_id != sample.part_id or prior.label != sample.label or prior.view != sample.view):
                    raise DatasetIntegrityError(f"Duplicate bytes claim incompatible labels/physical parts/views: {prior.path}, {path}")
                hashes[digest] = sample
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise DatasetIntegrityError(f"Invalid provenance metadata: {sidecar}: {exc}") from exc
            if (include_rejected or not (self.root / ".rejected" / f"{sample_id}.json").exists()) and (view is None or candidate_view == view):
                samples.append(sample)
        # Legacy flat directories cannot accidentally become a valid modern dataset.
        for label in ("GOOD", "NG"):
            if any(p.is_file() and p.suffix.lower() in EXTENSIONS for p in (self.root / label).glob("*")):
                raise DatasetIntegrityError(f"Legacy {label} images lack view/physical-part metadata; import them with explicit IDs")
        return samples

    def inventory(self, view: str | None = None) -> list[Sample]:
        return self._scan(view=view)

    def remove(self, sample_id: str, reason: str = "operator rejected sample") -> None:
        sample = next((s for s in self._scan(include_rejected=True) if s.id == sample_id), None)
        if sample is None:
            raise KeyError(f"Unknown dataset sample: {sample_id}")
        folder = self.root / ".rejected"
        folder.mkdir(exist_ok=True)
        target = folder / f"{sample.id}.json"
        if not target.exists():
            with target.open("x", encoding="utf-8") as stream:
                json.dump({"id": sample.id, "sha256": sample.sha256, "reason": reason,
                           "rejected_at": datetime.now(timezone.utc).isoformat()}, stream, indent=2)

    def stats(self) -> dict[str, Any]:
        all_samples = self._scan(include_rejected=True)
        active = self.inventory()
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
        for view, bucket in views.items():
            bucket["parts"] = len({s.part_id for s in active if s.view == view})
        part_labels = {s.part_id: max(x.label for x in active if x.part_id == s.part_id) for s in active}
        return {"images": len(active), "parts": len(part_labels),
                "good_parts": sum(v == 0 for v in part_labels.values()),
                "ng_parts": sum(v == 1 for v in part_labels.values()),
                "rejected": len(all_samples) - len(active), "views": views,
                "duplicates": {k: v for k, v in hashes.items() if len(v) > 1},
                "near_duplicate_warnings": near,
                "ng_sensitivity_validated": False}


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
    good = sorted(key for key, values in groups.items() if not any(s.label for s in values))
    ng = sorted(key for key, values in groups.items() if any(s.label for s in values))
    if len(good) < 3:
        raise ValueError("At least 3 independent GOOD physical parts/groups are required for train, validation and untouched test partitions")
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
