"""Atomic, append-only inspection evidence with lossless native-frame storage."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
from uuid import UUID

import cv2
import numpy as np

from src.inspection.session import InspectionSession
from src.utils.filesystem import display_path, filesystem_path


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return display_path(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _frame_digest(frame: np.ndarray) -> str:
    return sha256(_canonical({"shape": list(frame.shape), "dtype": str(frame.dtype)}) +
                  np.ascontiguousarray(frame).tobytes()).hexdigest()


def _validate_frame(frame: np.ndarray) -> None:
    if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or
            frame.shape[0] < 1 or frame.shape[1] < 1 or frame.dtype not in (np.uint8, np.uint16)):
        raise ValueError("Evidence must be a nonempty native BGR uint8/uint16 image")


def _write_file(path: Path, payload: bytes) -> None:
    with filesystem_path(path).open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


class EvidenceStore:
    def __init__(self, root: str | Path):
        self.root = filesystem_path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, session: InspectionSession, frames: dict[str, np.ndarray],
             marked: dict[str, np.ndarray] | None = None) -> Path:
        """Commit one finalized/cancelled event; verified retries return its path.

        No frame is resized, recompressed as JPEG, or inferred from a preview.
        Caller-provided originals must be the actual acquisition frames. Missing
        frames for recorded views are an error rather than fabricated evidence.
        """
        if session.final_decision is None:
            raise ValueError("Finalize or cancel the session before saving evidence")
        try:
            event_id = str(UUID(session.event_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("Evidence event requires a UUID identity") from exc
        missing = set(session.decisions) - set(frames)
        if missing:
            raise ValueError("Original evidence missing for recorded views: " + ", ".join(sorted(missing)))
        marked = marked or {}
        if set(marked) - set(frames):
            raise ValueError("Marked evidence requires the matching original frame")
        for view, frame in frames.items():
            if not re.fullmatch(r"[A-Za-z0-9_-]+", view):
                raise ValueError("View identities in evidence must contain letters, digits, underscores or hyphens")
            _validate_frame(frame)
        for view, frame in marked.items():
            _validate_frame(frame)
            if frame.shape != frames[view].shape or frame.dtype != frames[view].dtype:
                raise ValueError("Marked evidence must retain the original frame dimensions and dtype")
        snapshot = _jsonable(session.to_dict())
        source_fingerprint = sha256(_canonical({
            "session": snapshot,
            "frames": {v: _frame_digest(f) for v, f in sorted(frames.items())},
            "marked": {v: _frame_digest(f) for v, f in sorted(marked.items())},
        })).hexdigest()
        destination = self.root / event_id
        if destination.exists():
            self._verify_existing(destination, source_fingerprint)
            return destination

        lock = self.root / f".{event_id}.lock"
        try:
            lock_descriptor = os.open(str(lock), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise RuntimeError("This event is already being saved; retry after the active save finishes") from exc
        temporary: Path | None = None
        try:
            os.close(lock_descriptor)
            if destination.exists():
                self._verify_existing(destination, source_fingerprint)
                return destination
            temporary = Path(tempfile.mkdtemp(prefix=f".{event_id}-", dir=self.root))
            images = {}
            for view, original in sorted(frames.items()):
                images[view] = {"original": self._save_png(temporary, f"{view}_original.png", original)}
                if view in marked:
                    images[view]["marked"] = self._save_png(temporary, f"{view}_marked.png", marked[view])
            event = {"schema_version": 1, **snapshot, "images": images,
                     "source_fingerprint": source_fingerprint,
                     "storage": "lossless PNG; original acquisition dimensions and BGR pixel values"}
            _write_file(temporary / "event.json", _canonical(event))
            _write_file(temporary / "event.sha256", sha256((temporary / "event.json").read_bytes()).hexdigest().encode("ascii"))
            self.read(temporary)
            # Both directories share a parent/filesystem; the exclusive event
            # lock prevents cooperating writers from replacing another event.
            if destination.exists():
                self._verify_existing(destination, source_fingerprint)
                return destination
            os.rename(temporary, destination)
            temporary = None
            self._verify_existing(destination, source_fingerprint)
            return destination
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)
            lock.unlink(missing_ok=True)

    @staticmethod
    def _save_png(folder: Path, name: str, frame: np.ndarray) -> dict[str, Any]:
        success, encoded = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not success:
            raise OSError("OpenCV could not encode lossless evidence")
        payload = encoded.tobytes()
        _write_file(folder / name, payload)
        return {"path": name, "sha256": sha256(payload).hexdigest(),
                "pixel_sha256": _frame_digest(frame), "shape": list(frame.shape),
                "dtype": str(frame.dtype), "encoding": "PNG", "color_order": "BGR"}

    def _verify_existing(self, folder: Path, fingerprint: str) -> None:
        event = self.read(folder)
        if event.get("source_fingerprint") != fingerprint:
            raise FileExistsError("Event UUID already exists with different content; evidence was not overwritten")

    def read(self, folder: str | Path) -> dict[str, Any]:
        """Read an actual stored event and verify JSON, file and pixel integrity."""
        folder = filesystem_path(folder)
        event_file = folder / "event.json"
        raw = event_file.read_bytes()
        expected = (folder / "event.sha256").read_text("ascii").strip()
        if sha256(raw).hexdigest() != expected:
            raise ValueError("Inspection metadata integrity check failed")
        event = json.loads(raw)
        if not isinstance(event, dict) or event.get("schema_version") != 1:
            raise ValueError("Unsupported inspection evidence schema")
        try:
            event_id = str(UUID(str(event["event_id"])))
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError("Invalid inspection event identity") from exc
        try:
            directory_id = str(UUID(folder.name))
        except ValueError:
            directory_id = None  # Private staging directories are verified before publication.
        if directory_id is not None and event_id != directory_id:
            raise ValueError("Inspection event identity does not match its evidence directory")
        if (not isinstance(event.get("images"), dict) or not isinstance(event.get("views"), dict) or
                not isinstance(event.get("attempts"), list) or not isinstance(event.get("counted"), bool) or
                not isinstance(event.get("final_decision"), dict) or
                event["final_decision"].get("verdict") not in {"PASS", "FAIL", "RECHECK"}):
            raise ValueError("Incomplete inspection evidence metadata")
        if set(event["views"]) - set(event["images"]):
            raise ValueError("A captured view has no original evidence")
        for view, variants in event["images"].items():
            if not isinstance(variants, dict) or not isinstance(variants.get("original"), dict):
                raise ValueError("Original evidence metadata is missing")
            for variant, metadata in variants.items():
                if not isinstance(metadata, dict):
                    raise ValueError("Invalid image evidence metadata")
                name = metadata["path"]
                if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
                    raise ValueError("Unsafe evidence image path")
                path = folder / name
                if path.is_symlink():
                    raise ValueError("Evidence images cannot be symbolic links")
                payload = path.read_bytes()
                if sha256(payload).hexdigest() != metadata["sha256"]:
                    raise ValueError(f"Evidence file integrity check failed: {view}/{variant}")
                frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
                if frame is None or list(frame.shape) != metadata["shape"] or str(frame.dtype) != metadata["dtype"]:
                    raise ValueError(f"Evidence image dimensions or type changed: {view}/{variant}")
                if _frame_digest(frame) != metadata["pixel_sha256"]:
                    raise ValueError(f"Evidence pixel integrity check failed: {view}/{variant}")
        event["integrity_verified"] = True
        event["evidence_path"] = display_path(folder)
        return event

    def history(self) -> list[dict[str, Any]]:
        """Actual stored events only, including explicit corruption diagnostics."""
        events = []
        for folder in self.root.iterdir():
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            try:
                UUID(folder.name)
            except ValueError:
                continue
            try:
                events.append(self.read(folder))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                events.append({"event_id": folder.name, "evidence_path": display_path(folder),
                               "integrity_verified": False, "storage_error": str(exc)})
        return sorted(events, key=lambda row: row.get("created_at", ""), reverse=True)
