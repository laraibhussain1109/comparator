"""Independent-part diagnostics describe software fixtures, not model accuracy."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.data import DatasetIntegrityError, DatasetStore, Sample, split_by_part, training_readiness


def sample(index, part_id, label=0, view="inner_face", session=""):
    return Sample(str(index), Path(f"image-{index}.png"), label, view, part_id,
                  hashlib.sha256(str(index).encode()).hexdigest(), {"session_id": session})


def test_781_images_of_one_mixed_label_housing_explain_zero_good_parts():
    captures = [sample(index, "LEMON", label=int(index == 780)) for index in range(781)]
    report = training_readiness(captures)

    assert {key: report[key] for key in ("images", "good_images", "ng_images", "parts",
                                        "good_parts", "ng_parts", "mixed_label_parts",
                                        "eligible_good_groups", "eligible_ng_groups")} == {
        "images": 781, "good_images": 780, "ng_images": 1, "parts": 1,
        "good_parts": 0, "ng_parts": 1, "mixed_label_parts": 1,
        "eligible_good_groups": 0, "eligible_ng_groups": 1,
    }
    assert report["part_details"] == [{
        "part_id": "LEMON", "images": 781, "good_images": 780, "ng_images": 1,
        "label": "NG", "mixed_labels": True, "views": ["inner_face"], "session_ids": [],
    }]
    assert not report["can_split"]
    assert report["minimum_good_groups"] == 3
    assert "781 images" in report["message"]
    assert "LEMON" in report["message"]
    assert "1 mixed-label" in report["message"]
    assert "excludes all views" in report["message"]
    assert "never invent IDs" in report["message"]
    with pytest.raises(ValueError, match="At least 3 independent GOOD") as exc:
        split_by_part(captures)
    assert str(exc.value) == report["message"]


def test_781_good_captures_still_represent_only_one_independent_housing():
    captures = [sample(index, "LEMON") for index in range(781)]
    report = training_readiness(iter(captures))

    assert report["good_images"] == 781
    assert report["good_parts"] == report["eligible_good_groups"] == 1
    assert report["ng_parts"] == report["mixed_label_parts"] == 0
    assert not report["can_split"]
    assert "Repeated photos still count as one housing" in report["message"]
    assert "NG-labelled capture" not in report["message"]
    with pytest.raises(ValueError, match="Independent GOOD groups: 1"):
        split_by_part(captures)


def test_three_actual_good_housings_meet_requirement_without_cross_view_leakage():
    captures = [sample(index * 2 + offset, f"physical-{index}", view=view)
                for index in range(3)
                for offset, view in enumerate(("inner_face", "ribbed_face"))]
    report = training_readiness(captures)
    split = split_by_part(captures)
    memberships = [{s.part_id for s in getattr(split, name)}
                   for name in ("train", "validation", "test")]

    assert report["can_split"]
    assert report["eligible_good_groups"] == 3
    assert report["good_images"] == 6
    assert "Selected-face coverage and image quality" in report["message"]
    assert all(len(parts) == 1 for parts in memberships)
    assert len(set.union(*memberships)) == 3
    assert all(len(getattr(split, name)) == 2 for name in ("train", "validation", "test"))


def test_session_grouping_reports_three_good_parts_collapsing_to_one_group():
    captures = [sample(index, f"physical-{index}", session="day-one") for index in range(3)]
    ungrouped = training_readiness(captures)
    grouped = training_readiness(captures, group_sessions=True)

    assert ungrouped["can_split"]
    assert not ungrouped["group_sessions"]
    assert split_by_part(captures).train
    assert grouped["parts"] == grouped["good_parts"] == 3
    assert grouped["eligible_good_groups"] == 1
    assert grouped["group_sessions"]
    assert not grouped["can_split"]
    assert "3 physical housing(s) form 1 independent group(s)" in grouped["message"]
    assert "day-one" in grouped["message"]
    with pytest.raises(ValueError, match="Shared capture sessions"):
        split_by_part(captures, group_sessions=True)


def test_session_groups_are_transitive_and_ng_excludes_connected_good_parts():
    captures = [sample(0, "A", session="first"), sample(1, "B", session="first"),
                sample(2, "B", session="second"), sample(3, "C", label=1, session="second")]
    report = training_readiness(captures, group_sessions=True)

    assert report["good_parts"] == 2
    assert report["ng_parts"] == 1
    assert report["eligible_good_groups"] == 0
    assert report["eligible_ng_groups"] == 1
    assert "excludes its whole session group" in report["message"]
    with pytest.raises(ValueError, match="Independent GOOD groups: 0"):
        split_by_part(captures, group_sessions=True)


def test_ng_on_other_face_excludes_entire_part_but_preserves_three_good_groups():
    captures = [sample(index, f"good-{index}") for index in range(3)]
    captures += [sample(3, "mixed", view="inner_face"),
                 sample(4, "mixed", label=1, view="ribbed_face")]
    report = training_readiness(captures)
    split = split_by_part(captures)

    assert report["can_split"]
    assert report["good_parts"] == 3
    assert report["ng_parts"] == report["mixed_label_parts"] == 1
    assert not any(s.part_id == "mixed" for s in split.train)
    assert {s.view for s in split.test if s.part_id == "mixed"} == {"inner_face", "ribbed_face"}
    details = next(part for part in report["part_details"] if part["part_id"] == "mixed")
    assert details["good_images"] == details["ng_images"] == 1


def test_empty_dataset_explains_import_requirement():
    report = training_readiness([])

    assert not report["can_split"]
    assert report["part_details"] == []
    assert report["images"] == report["parts"] == report["eligible_good_groups"] == 0
    assert "Import captures of actual GOOD housings" in report["message"]
    with pytest.raises(ValueError, match="At least 3 independent GOOD"):
        split_by_part([])


@pytest.mark.parametrize("conflict", [
    {"part_id": "other-part"}, {"view": "ribbed_face"}, {"label": 1},
])
def test_readiness_and_split_reject_same_duplicate_identity_conflicts(conflict):
    original = sample(0, "actual-part")
    conflicting = Sample("copy", Path("copy.png"), conflict.get("label", original.label),
                         conflict.get("view", original.view), conflict.get("part_id", original.part_id),
                         original.sha256)

    for function in (training_readiness, split_by_part):
        with pytest.raises(DatasetIntegrityError, match="Duplicate image bytes"):
            function([original, conflicting])


def test_readiness_examples_are_bounded_but_part_details_remain_complete():
    captures = [sample(index, f"part-{index:02d}-" + "x" * 500, label=1) for index in range(30)]
    report = training_readiness(captures)

    assert len(report["part_details"]) == 30
    assert "and 28 more" in report["message"]
    assert len(report["message"]) < 850


def test_stats_exposes_group_aware_readiness_and_image_label_counts(tmp_path):
    store = DatasetStore(tmp_path)
    for index in range(3):
        image = np.random.default_rng(index).integers(0, 256, (20, 20, 3), dtype=np.uint8)
        store.capture_image(image, "inner_face", "GOOD", f"actual-{index}", "one-session")
    image = np.random.default_rng(7).integers(0, 256, (20, 20, 3), dtype=np.uint8)
    store.capture_image(image, "ribbed_face", "NG", "actual-2", "one-session")

    stats = store.stats()
    grouped = store.stats(group_sessions=True)
    assert stats["good_images"] == 3
    assert stats["ng_images"] == 1
    assert stats["good_parts"] == 2
    assert stats["ng_parts"] == 1
    assert stats["training_readiness"]["eligible_good_groups"] == 2
    assert grouped["training_readiness"]["eligible_good_groups"] == 0
    assert grouped["training_readiness"]["eligible_ng_groups"] == 1
