"""Importer repair -> real geometry-training integration using software fixtures.

The supplied synthetic housing IDs are explicitly test identities. These checks
establish metadata integrity and split behavior, not industrial model accuracy.
"""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from src.artifacts import ArtifactStore
from src.data import DatasetIntegrityError, split_by_part
from src.service import NeuroIrisService
from test_smoke_training import import_housings, make_service
from test_vision import housing


def sidecar(path):
    return path.with_suffix(path.suffix + ".json")


def partition_ids(samples):
    split = split_by_part(samples)
    result = {name: {sample.part_id for sample in getattr(split, name)}
              for name in ("train", "validation", "test")}
    assert not result["train"] & result["validation"]
    assert not result["train"] & result["test"]
    assert not result["validation"] & result["test"]
    return result


def unbacked_empty_sidecar(service, sample):
    path = Path(sample["path"])
    service.dataset._mirror(path).unlink()
    sidecar(path).write_bytes(b"")
    return path


def assert_geometry_training_report(summary, expected_partitions):
    assert summary["backend"] == "geometry"
    assert summary["test"]["count"] > 0
    assert summary["report"]["records"]
    assert all(record["error"] is None for record in summary["report"]["records"])
    metadata = ArtifactStore.verify(Path(summary["artifact_path"]))
    assert set(metadata["training_parts"]) == expected_partitions["train"]
    assert set(metadata["validation_parts"]) == expected_partitions["validation"]
    assert set(metadata["test_parts"]) == expected_partitions["test"]


def test_verified_metadata_mirror_recovers_empty_sidecar_before_geometry_training(tmp_path):
    service = make_service(tmp_path)
    import_housings(service)
    original_samples = service.samples()
    expected_partitions = partition_ids(service.dataset.inventory())
    for sample in original_samples:
        path = Path(sample["path"])
        wrapper = json.loads(service.dataset._mirror(path).read_text(encoding="utf-8"))
        assert wrapper["metadata"] == json.loads(sidecar(path).read_text(encoding="utf-8"))
        assert wrapper["metadata"]["part_id"] == sample["part_id"]
        assert wrapper["image_sha256"] == sha256(path.read_bytes()).hexdigest()
    sample = next(row for row in original_samples if row["view"] == "ribbed_face")
    path = Path(sample["path"])
    native_bytes = path.read_bytes()
    sidecar(path).write_bytes(b"")

    restarted = NeuroIrisService(service.config)
    summary = restarted.train("ribbed_face", "geometry")

    assert_geometry_training_report(summary, expected_partitions)
    assert path.read_bytes() == native_bytes
    assert json.loads(sidecar(path).read_text(encoding="utf-8")) == sample["metadata"]
    assert restarted.pending_samples() == []
    assert len(restarted.samples()) == len(original_samples)
    archived = list((restarted.dataset.root / ".metadata" / "repairs").glob("*.bin"))
    assert archived and any(backup.read_bytes() == b"" for backup in archived)
    report = restarted.evaluate("ribbed_face", "geometry")
    assert report["records"] and all(record["error"] is None for record in report["records"])


def test_unrecoverable_old_sidecar_is_visible_and_explicit_repair_preserves_source(tmp_path):
    service = make_service(tmp_path)
    import_housings(service)
    expected_partitions = partition_ids(service.dataset.inventory())
    original_samples = service.samples()
    sample = next(row for row in original_samples if row["view"] == "ribbed_face")
    old_path = unbacked_empty_sidecar(service, sample)
    native_bytes = old_path.read_bytes()

    restarted = NeuroIrisService(service.config)
    assert len(restarted.samples()) == len(original_samples) - 1
    pending = restarted.pending_samples()
    assert len(pending) == 1 and Path(pending[0]["path"]) == old_path
    assert pending[0]["status"] == "repair_required"
    assert restarted.dataset_stats()["unmanaged_count"] == 1
    with pytest.raises(DatasetIntegrityError) as captured:
        restarted.train("ribbed_face", "geometry")
    assert "Register / Repair Selected" in str(captured.value)
    assert "Expecting value" not in str(captured.value)
    assert old_path.read_bytes() == native_bytes and sidecar(old_path).read_bytes() == b""

    repaired = restarted.repair_sample(old_path, sample["view"], sample["label"],
                                      sample["part_id"], sample["metadata"]["session_id"])
    repaired_path = Path(repaired["path"])
    backup = restarted.dataset.root / repaired["metadata"]["metadata_repair_backup"]
    assert repaired_path.read_bytes() == native_bytes
    assert repaired["sha256"] == sha256(native_bytes).hexdigest()
    assert backup.is_file() and backup.read_bytes() == b""
    assert repaired["part_id"] == sample["part_id"]
    assert repaired["metadata"]["session_id"] == sample["metadata"]["session_id"]
    assert restarted.pending_samples() == []
    assert len(restarted.samples()) == len(original_samples)
    assert partition_ids(restarted.dataset.inventory()) == expected_partitions

    summary = restarted.train("ribbed_face", "geometry")
    assert_geometry_training_report(summary, expected_partitions)


def test_repair_never_turns_photographs_of_one_part_into_independent_parts(tmp_path):
    service = make_service(tmp_path)
    captures = []
    for index in range(4):
        image = housing(True)
        image[0, index] = np.array([index + 1, 3, 7], np.uint8)
        captures.append(service.capture_sample(image, "ribbed_face", "GOOD", "one-actual-test-part", "one-placement"))
    original = captures[0]
    path = unbacked_empty_sidecar(service, original)
    service.repair_sample(path, "ribbed_face", "GOOD", "one-actual-test-part", "one-placement")

    samples = service.samples()
    assert len(samples) == 4
    assert {sample["part_id"] for sample in samples} == {"one-actual-test-part"}
    assert service.dataset_stats()["parts"] == 1
    with pytest.raises(ValueError, match="3 independent GOOD"):
        service.train("ribbed_face", "geometry")


def test_session_grouping_does_not_silently_split_one_capture_group(tmp_path):
    service = make_service(tmp_path)
    import_housings(service)
    # The helper deliberately records every capture in the same fixture session.
    service.config["dataset"] = {"group_sessions": True, "split": [.7, .15, .15]}
    assert service.dataset_stats()["parts"] == 7
    with pytest.raises(ValueError, match="3 independent GOOD"):
        service.train("ribbed_face", "geometry")
