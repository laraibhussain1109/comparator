"""Long/Unicode dataset paths: real filesystem operations, including native Windows CI."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data import DatasetIntegrityError, DatasetStore
from src.utils.filesystem import display_path, filesystem_path


def deep_path(base: Path, leaf: str) -> Path:
    parts = [f"downloaded-inspection-{index:02d}-鋁件検査-repeated-folder" for index in range(7)]
    path = filesystem_path(base.joinpath(*parts, leaf))
    assert len(display_path(path)) > 260
    return path


def photo(path: Path, seed: int = 1) -> bytes:
    path = filesystem_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.random.default_rng(seed).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(path.suffix, frame)
    assert success
    data = encoded.tobytes()
    path.write_bytes(data)
    return data


def sidecar(path: Path) -> Path:
    path = filesystem_path(path)
    return path.with_suffix(path.suffix + ".json")


def assert_display(value: str) -> None:
    assert not value.startswith("\\\\?\\")
    assert not value.lower().startswith("\\\\?\\unc\\")


def test_deep_unicode_external_import_and_display_path_roundtrip_are_idempotent(tmp_path):
    store = DatasetStore(display_path(deep_path(tmp_path, "dataset")))
    source = deep_path(tmp_path, "camera-source") / "相機照片_原件.png"
    original = photo(source)
    if os.name == "nt":
        assert str(store.root).startswith("\\\\?\\")
    first = store.import_image(display_path(source), "inner_face", "GOOD", "Lemon-鋁殼-01", "十月")
    second = store.import_image(source, "inner_face", "GOOD", "Lemon-鋁殼-01", "later-session")
    assert first == second
    assert first.session_id == "十月"
    assert first.path.read_bytes() == source.read_bytes() == original
    assert first.metadata["original_filename"] == "相機照片_原件.png"
    assert first.metadata["source"] == display_path(filesystem_path(source).resolve())
    shown = first.to_dict()["path"]
    assert_display(shown)
    assert filesystem_path(shown).read_bytes() == original
    assert first.part_id == "Lemon-鋁殼-01"
    assert store.inventory() == [first]


def test_deep_internal_pending_selection_uses_plain_and_extended_paths_consistently(tmp_path):
    store = DatasetStore(deep_path(tmp_path, "dataset"))
    selected = store.root / "inner_face" / "GOOD" / "選択_写真.png"
    remaining = store.root / "inner_face" / "GOOD" / "残り_写真.png"
    original = photo(selected, 1)
    photo(remaining, 2)
    assert store._internal(Path(display_path(selected)))
    assert store._internal(filesystem_path(selected))
    same_root = DatasetStore(display_path(store.root))
    assert same_root._lock is store._lock
    pending = store.unmanaged_images()
    assert {r["path"] for r in pending} == {display_path(selected), display_path(remaining)}
    for row in pending:
        assert_display(row["path"])
        assert filesystem_path(row["path"]).is_file()
    sample = store.import_image(display_path(selected), "inner_face", "GOOD", "Lemon", "oct")
    assert not selected.exists()
    assert sample.path.read_bytes() == original
    assert store.inventory(strict=False) == [sample]
    assert store.stats()["unmanaged_count"] == 1
    assert store.unmanaged_images()[0]["path"] == display_path(remaining)
    assert store._mirror(Path(sample.to_dict()["path"])) == store._mirror(sample.path)
    wrapper = json.loads(store._mirror(sample.path).read_text(encoding="utf-8"))
    expected_relative = sample.path.relative_to(store.root).as_posix()
    assert wrapper["relative_path"] == expected_relative
    assert store._mirror(sample.path).stem == hashlib.sha256(expected_relative.encode("utf-8")).hexdigest()
    with pytest.raises(DatasetIntegrityError, match="Missing metadata/physical part ID"):
        same_root.inventory()


def test_deep_unicode_explicit_repair_and_trusted_recovery_preserve_provenance(tmp_path):
    store = DatasetStore(display_path(deep_path(tmp_path, "dataset")))
    broken = store.root / "inner_face" / "GOOD" / "WIN_原始照片.png"
    original = photo(broken)
    invalid_metadata = b"{unfinished sidecar"
    sidecar(broken).write_bytes(invalid_metadata)
    pending = store.unmanaged_images()[0]
    assert pending["status"] == "repair_required"
    assert pending["path"] == display_path(broken)
    assert_display(pending["path"])
    sample = store.repair_image(pending["path"], "inner_face", "GOOD", "実物-Lemon", "oct")
    assert sample.path.read_bytes() == original
    assert not broken.exists() and not sidecar(broken).exists()
    backup = filesystem_path(store.root / sample.metadata["metadata_repair_backup"])
    assert backup.read_bytes() == invalid_metadata
    audit = json.loads(backup.with_suffix(".json").read_text(encoding="utf-8"))
    assert audit["original_image"] == display_path(broken)
    assert audit["selection"]["part_id"] == "実物-Lemon"
    assert sample.metadata["repaired_from"] == display_path(broken)
    assert_display(audit["original_image"])
    assert_display(audit["original_sidecar"])
    # A plain UI path must not turn an intact record into a second physical sample.
    repeated = store.repair_image(sample.to_dict()["path"], "inner_face", "GOOD", "実物-Lemon", "other-session")
    assert repeated == sample and repeated.session_id == "oct"
    original_sidecar = sidecar(sample.path).read_bytes()
    sidecar(sample.path).write_bytes(b"")
    reopened = DatasetStore(display_path(store.root))
    assert reopened.inventory() == [sample]
    assert sidecar(sample.path).read_bytes() == original_sidecar
    assert sample.path.read_bytes() == original


def test_deep_path_normalization_preserves_hash_conflicts_and_external_sources(tmp_path):
    store = DatasetStore(deep_path(tmp_path, "dataset"))
    external = deep_path(tmp_path, "source") / "原件.png"
    original = photo(external)
    sample = store.import_image(display_path(external), "inner_face", "GOOD", "real-part")
    with pytest.raises(DatasetIntegrityError, match="conflict"):
        store.import_image(display_path(external), "inner_face", "NG", "invented-part")
    assert external.read_bytes() == original
    replacement = deep_path(tmp_path, "replacement") / "other.png"
    sample.path.write_bytes(photo(replacement, 99))
    with pytest.raises(DatasetIntegrityError, match="immutable sidecar"):
        DatasetStore(display_path(store.root)).inventory(strict=False)
    assert external.read_bytes() == original
