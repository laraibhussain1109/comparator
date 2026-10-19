"""Regression coverage for manually copied photos and interrupted metadata writes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import src.data.dataset as dataset_module
from src.data import DatasetIntegrityError, DatasetStore


def photo(path: Path, seed: int = 1) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.random.default_rng(seed).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path.read_bytes()


def raw(store: DatasetStore, name="WIN_20261018_12_00_00_Pro.jpg", seed=1):
    path = store.root / "inner_face" / "GOOD" / name
    return path, photo(path, seed)


def sidecar(path):
    return path.with_suffix(path.suffix + ".json")


def test_register_selected_raw_from_screenshot_uses_selections_and_preserves_bytes(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    selected, original = raw(store)
    sample = store.import_image(selected, "inner_face", "GOOD", "Lemon", "oct")
    assert sample.path.name.startswith("Lemon_inner_face_GOOD_oct_")
    assert sample.path.name.endswith(".jpg")
    assert sample.path.read_bytes() == original
    assert sample.metadata["original_filename"] == selected.name
    assert sample.part_id == "Lemon" and sample.session_id == "oct"
    assert not selected.exists()
    assert store.inventory() == [sample]
    assert store.stats()["unmanaged_count"] == 0


def test_external_sources_preserved_and_same_identity_import_is_idempotent(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    source = tmp_path / "camera-original.JPG"
    original = photo(source)
    first = store.import_image(source, "inner_face", "GOOD", "actual-part", "oct")
    second = store.import_image(source, "inner_face", "GOOD", "actual-part", "nov")
    assert first == second
    assert second.session_id == "oct"
    assert source.read_bytes() == original
    assert len(store.inventory()) == 1


def test_unselected_raw_remains_pending_ui_tolerates_but_training_is_strict(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    first, _ = raw(store, "first.jpg", 1)
    second, original = raw(store, "second.jpg", 2)
    sample = store.import_image(first, "inner_face", "GOOD", "one", "oct")
    assert store.inventory(strict=False) == [sample]
    stats = store.stats()
    assert stats["images"] == stats["parts"] == 1
    assert stats["unmanaged_count"] == 1
    assert stats["unmanaged_images"][0]["path"] == str(second)
    assert stats["unmanaged_images"][0]["status"] == "unassigned"
    assert second.read_bytes() == original
    with pytest.raises(DatasetIntegrityError, match="Missing metadata/physical part ID"):
        store.inventory()


def test_legacy_raw_can_be_assigned_without_guessing_view_or_physical_identity(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    source = store.root / "GOOD" / "raw.jpg"
    original = photo(source)
    assert store.unmanaged_images()[0]["view"] is None
    sample = store.import_image(source, "ribbed_face", "GOOD", "physical-20")
    assert sample.path.read_bytes() == original
    assert sample.view == "ribbed_face" and sample.part_id == "physical-20"
    assert not source.exists()


def test_duplicate_raw_matching_managed_is_cleaned_but_conflicting_identity_is_not(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    source = tmp_path / "original.jpg"
    original = photo(source)
    existing = store.import_image(source, "inner_face", "GOOD", "same-part", "first-session")
    extra = store.root / "inner_face" / "GOOD" / "copied.jpg"
    extra.write_bytes(original)
    assert store.import_image(extra, "inner_face", "GOOD", "same-part", "new-session") == existing
    assert not extra.exists()
    extra.write_bytes(original)
    with pytest.raises(DatasetIntegrityError, match="conflict"):
        store.import_image(extra, "inner_face", "NG", "invented-new-part")
    assert extra.read_bytes() == original
    assert len(store.inventory(strict=False)) == 1


def test_windows_safe_bounded_names_preserve_exact_ids_in_metadata(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    source = tmp_path / "source.jpg"
    photo(source)
    identity = 'CON:<bad>"|?*\\name/' + "very-long" * 50
    session = 'oct:11/20\\*" ?' * 30
    sample = store.import_image(source, "inner_face", "GOOD", identity, session)
    assert len(sample.path.name) < 90
    assert not any(c in sample.path.name for c in '<>:"/\\|?*')
    assert not sample.path.name.endswith((" ", "."))
    assert sample.part_id == identity and sample.session_id == session


def test_atomic_sidecar_failure_preserves_source_and_removes_only_created_files(tmp_path, monkeypatch):
    store = DatasetStore(tmp_path / "dataset")
    source, original = raw(store)
    atomic = dataset_module._atomic_bytes
    def fail_final(path, payload, **kwargs):
        if path.name.endswith(".jpg.json"):
            raise OSError("simulated sidecar disk failure")
        return atomic(path, payload, **kwargs)
    monkeypatch.setattr(dataset_module, "_atomic_bytes", fail_final)
    with pytest.raises(OSError, match="disk failure"):
        store.import_image(source, "inner_face", "GOOD", "Lemon", "oct")
    assert source.read_bytes() == original
    assert store.inventory(strict=False) == []
    assert store.unmanaged_images()[0]["path"] == str(source)
    assert not list((store.root / ".metadata").glob("*.json"))
    assert not list(store.root.rglob("*.tmp"))


def test_exclusive_target_collision_does_not_delete_preexisting_user_file(tmp_path, monkeypatch):
    store = DatasetStore(tmp_path / "dataset")
    source, original = raw(store)
    fixed_id = "1" * 32
    existing = source.parent / ("Lemon_inner_face_GOOD_oct_" + fixed_id[:12] + ".jpg")
    other = photo(existing, 2)
    monkeypatch.setattr(dataset_module.uuid, "uuid4", lambda: SimpleNamespace(hex=fixed_id))
    with pytest.raises(FileExistsError):
        store.import_image(source, "inner_face", "GOOD", "Lemon", "oct")
    assert source.read_bytes() == original
    assert existing.read_bytes() == other


def test_cleanup_failure_retains_verified_copy_and_retry_is_safe(tmp_path, monkeypatch):
    store = DatasetStore(tmp_path / "dataset")
    source, original = raw(store)
    unlink = Path.unlink
    def locked(path, *args, **kwargs):
        if path == source:
            raise PermissionError("photo is open in another program")
        return unlink(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", locked)
        with pytest.raises(DatasetIntegrityError, match="Verified managed copy retained"):
            store.import_image(source, "inner_face", "GOOD", "Lemon", "oct")
    saved = store.inventory(strict=False)[0]
    assert source.read_bytes() == saved.path.read_bytes() == original
    reused = store.import_image(source, "inner_face", "GOOD", "Lemon", "nov")
    assert reused.id == saved.id and reused.session_id == "oct"
    assert not source.exists()
    assert store.inventory() == [saved]


def test_empty_or_missing_sidecar_recovers_only_from_matching_durable_mirror(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    source = tmp_path / "source.jpg"
    original = photo(source)
    sample = store.import_image(source, "inner_face", "GOOD", "Lemon", "oct")
    expected = sidecar(sample.path).read_bytes()
    sidecar(sample.path).write_bytes(b"")
    assert store.inventory()[0] == sample
    assert sidecar(sample.path).read_bytes() == expected
    assert any(p.read_bytes() == b"" for p in (store.root / ".metadata" / "repairs").glob("*.bin"))
    sidecar(sample.path).unlink()
    assert store.inventory()[0] == sample
    assert sample.path.read_bytes() == original


def test_untrusted_corrupt_sidecar_is_pending_and_requires_explicit_identity(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    broken, original = raw(store)
    sidecar(broken).write_bytes(b"")
    assert store.inventory(strict=False) == []
    pending = store.unmanaged_images()[0]
    assert pending["status"] == "repair_required"
    assert "Invalid provenance metadata" in pending["reason"]
    with pytest.raises(DatasetIntegrityError, match="Invalid provenance metadata"):
        store.inventory()
    sample = store.repair_image(broken, "inner_face", "GOOD", "real-part", "oct")
    assert sample.part_id == "real-part" and sample.path.read_bytes() == original
    backup = store.root / sample.metadata["metadata_repair_backup"]
    assert backup.read_bytes() == b""
    audit = json.loads(backup.with_suffix(".json").read_text())
    assert audit["selection"]["part_id"] == "real-part"
    assert audit["sidecar_sha256"] == hashlib.sha256(b"").hexdigest()
    assert not broken.exists() and not sidecar(broken).exists()
    assert store.inventory() == [sample]


def test_import_explicit_selected_corruption_repairs_but_unrelated_corruption_stays_pending(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    selected, original = raw(store, "selected.jpg", 1)
    unrelated, _ = raw(store, "unrelated.jpg", 2)
    sidecar(selected).write_bytes(b"{truncated")
    sidecar(unrelated).write_bytes(b"")
    sample = store.import_image(selected, "inner_face", "GOOD", "Lemon", "oct")
    assert sample.path.read_bytes() == original
    assert sidecar(unrelated).read_bytes() == b""
    assert len(store.inventory(strict=False)) == 1
    assert store.stats()["unmanaged_count"] == 1
    with pytest.raises(DatasetIntegrityError, match="Invalid provenance metadata"):
        store.inventory()


def test_hash_label_or_mirror_identity_conflicts_are_never_quietly_repaired(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    source = tmp_path / "original.jpg"
    photo(source)
    sample = store.import_image(source, "inner_face", "GOOD", "actual-part")
    metadata = sample.metadata.copy()
    metadata["part_id"] = "changed-part"
    sidecar(sample.path).write_text(json.dumps(metadata))
    with pytest.raises(DatasetIntegrityError, match="immutable metadata recovery mirror"):
        store.inventory(strict=False)
    with pytest.raises(DatasetIntegrityError, match="immutable metadata recovery mirror"):
        store.repair_image(sample.path, "inner_face", "GOOD", "changed-part")
    sidecar(sample.path).write_text(json.dumps({**sample.metadata, "label": 1}))
    with pytest.raises(DatasetIntegrityError, match="label differs"):
        store.repair_image(sample.path, "inner_face", "GOOD", "actual-part")
    sidecar(sample.path).write_bytes(b"")
    sample.path.write_bytes(photo(tmp_path / "replacement.jpg", 22))
    with pytest.raises(DatasetIntegrityError, match="recovery mirror"):
        store.inventory(strict=False)


def test_partial_identity_evidence_cannot_be_overwritten_by_selection(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    source, original = raw(store)
    sidecar(source).write_text(json.dumps({"part_id": "known-original", "sha256": hashlib.sha256(original).hexdigest()}))
    with pytest.raises(DatasetIntegrityError, match="conflicts with the selected identity"):
        store.repair_image(source, "inner_face", "GOOD", "invented-part")
    assert source.read_bytes() == original


def test_repair_cannot_resurrect_rejected_image_after_metadata_loss(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    source = tmp_path / "original.jpg"
    photo(source)
    sample = store.import_image(source, "inner_face", "GOOD", "actual-part")
    store.remove(sample.id)
    store._mirror(sample.path).unlink()
    sidecar(sample.path).write_bytes(b"")
    with pytest.raises(DatasetIntegrityError, match="cannot silently restore"):
        store.repair_image(sample.path, "inner_face", "GOOD", "actual-part")
    assert store.inventory(strict=False) == []


def test_metadata_publication_cannot_clobber_existing_sidecar(tmp_path):
    target = tmp_path / "metadata.json"
    target.write_bytes(b"existing-user-data")
    with pytest.raises(FileExistsError):
        dataset_module._atomic_bytes(target, b"new-data")
    assert target.read_bytes() == b"existing-user-data"
    assert not list(tmp_path.glob("*.tmp"))


def test_image_is_published_only_after_complete_metadata_and_root_locks_are_shared(tmp_path, monkeypatch):
    store = DatasetStore(tmp_path / "dataset")
    another = DatasetStore(store.root)
    assert another._lock is store._lock
    source = tmp_path / "source.jpg"
    photo(source)
    real_link = dataset_module.os.link
    observed = []
    def link_after_metadata(source_path, target, *args, **kwargs):
        target = Path(target)
        if target.suffix.lower() == ".jpg":
            assert not target.exists()
            details = json.loads(sidecar(target).read_text())
            wrapper = json.loads(store._mirror(target).read_text())
            assert wrapper["metadata"] == details
            observed.append(target)
        return real_link(source_path, target, *args, **kwargs)
    monkeypatch.setattr(dataset_module.os, "link", link_after_metadata)
    sample = store.import_image(source, "inner_face", "GOOD", "Lemon", "oct")
    assert observed == [sample.path]
    assert another.inventory() == [sample]


def test_windows_bom_sidecar_and_mirror_are_valid_without_repair(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    source = tmp_path / "source.jpg"
    photo(source)
    sample = store.import_image(source, "inner_face", "GOOD", "Lemon", "oct")
    for path in (sidecar(sample.path), store._mirror(sample.path)):
        path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert store.inventory() == [sample]
    assert store.unmanaged_images() == []
    assert not (store.root / ".metadata" / "repairs").exists()


def test_explicit_repair_write_failure_keeps_source_and_original_bad_sidecar(tmp_path, monkeypatch):
    store = DatasetStore(tmp_path / "dataset")
    source, original = raw(store)
    broken = b"{truncated original"
    sidecar(source).write_bytes(broken)
    atomic = dataset_module._atomic_bytes
    def fail_new_sidecar(path, payload, **kwargs):
        if path.name.endswith(".jpg.json") and path != sidecar(source):
            raise OSError("write interrupted")
        return atomic(path, payload, **kwargs)
    monkeypatch.setattr(dataset_module, "_atomic_bytes", fail_new_sidecar)
    with pytest.raises(OSError, match="write interrupted"):
        store.repair_image(source, "inner_face", "GOOD", "Lemon", "oct")
    assert source.read_bytes() == original
    assert sidecar(source).read_bytes() == broken
    assert any(p.read_bytes() == broken for p in (store.root / ".metadata" / "repairs").glob("*.bin"))
    assert store.inventory(strict=False) == []
    assert not list(store.root.rglob("*.tmp"))


def test_symlink_sources_and_metadata_cannot_escape_dataset(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    outside = tmp_path / "outside.jpg"
    original = photo(outside)
    link = store.root / "inner_face" / "GOOD" / "link.jpg"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symbolic links are unavailable on this Windows configuration")
    with pytest.raises(DatasetIntegrityError, match="symbolic link"):
        store.import_image(link, "inner_face", "GOOD", "Lemon")
    assert outside.read_bytes() == original


def test_windows_publication_uses_nonreplacing_rename_without_hardlink(tmp_path, monkeypatch):
    staged = tmp_path / ".image-staged.tmp"
    target = tmp_path / "photo.jpg"
    staged.write_bytes(b"new-image")
    target.write_bytes(b"existing-original")
    real_replace = dataset_module.os.replace
    calls = []
    def windows_rename(source, destination):
        calls.append((source, destination))
        # Reproduce documented Windows rename semantics on this POSIX test host.
        if Path(destination).exists():
            raise FileExistsError("Windows rename refuses an existing destination")
        real_replace(source, destination)
    def no_hardlink(*args, **kwargs):
        raise AssertionError("Windows publication must not require hard-link support")
    monkeypatch.setattr(dataset_module, "_WINDOWS", True)
    monkeypatch.setattr(dataset_module.os, "rename", windows_rename)
    monkeypatch.setattr(dataset_module.os, "link", no_hardlink)
    with pytest.raises(FileExistsError):
        dataset_module._publish_new(staged, target)
    assert staged.read_bytes() == b"new-image"
    assert target.read_bytes() == b"existing-original"
    target.unlink()
    dataset_module._publish_new(staged, target)
    assert target.read_bytes() == b"new-image"
    assert not staged.exists()
    assert len(calls) == 2


def test_windows_atomic_metadata_and_image_import_need_no_hardlinks(tmp_path, monkeypatch):
    real_replace = dataset_module.os.replace
    def windows_rename(source, destination):
        if Path(destination).exists():
            raise FileExistsError("Windows destination exists")
        real_replace(source, destination)
    def no_hardlink(*args, **kwargs):
        raise AssertionError("Hard links unavailable on this simulated Windows volume")
    monkeypatch.setattr(dataset_module, "_WINDOWS", True)
    monkeypatch.setattr(dataset_module.os, "rename", windows_rename)
    monkeypatch.setattr(dataset_module.os, "link", no_hardlink)
    store = DatasetStore(tmp_path / "dataset")
    source, original = raw(store)
    sample = store.import_image(source, "inner_face", "GOOD", "Lemon", "oct")
    assert sample.path.read_bytes() == original
    assert store.inventory() == [sample]
    assert not source.exists()
