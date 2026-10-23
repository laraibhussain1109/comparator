"""Long Unicode paths exercise Python filesystem I/O, never OpenCV filenames."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.inspection.evidence import EvidenceStore
from src.inspection.session import InspectionSession, ViewDecision
from src.utils.filesystem import display_path, filesystem_path
from src.utils.image_utils import read_image, write_image
from src.vision import GeometryProfile
from test_vision import housing


def long_unicode_root(tmp_path):
    root = tmp_path
    for index in range(9):
        root /= f"铝件-हाउसिंग-{index}-inspection-records"
    assert len(str(root)) > 300
    filesystem_path(root).mkdir(parents=True, exist_ok=True)
    return root


def forbid_filename_codec(*args, **kwargs):
    raise AssertionError("Image filenames must be handled by Python filesystem I/O")


def test_long_unicode_image_roundtrip_uses_memory_codecs_and_preserves_source(tmp_path, monkeypatch):
    root = long_unicode_root(tmp_path)
    monkeypatch.setattr(cv2, "imread", forbid_filename_codec)
    monkeypatch.setattr(cv2, "imwrite", forbid_filename_codec)
    monkeypatch.setattr(np, "fromfile", forbid_filename_codec)
    pixels = np.arange(73 * 101 * 3, dtype=np.uint16).reshape(73, 101, 3).astype(np.uint8)
    source = root / "参考原图-नमूना.PNG"
    destination = root / "saved" / "训练参考图.png"
    write_image(source, pixels)
    original_bytes = filesystem_path(source).read_bytes()

    decoded = read_image(source)
    write_image(destination, decoded)

    assert np.array_equal(decoded, pixels)
    assert np.array_equal(read_image(destination), pixels)
    assert filesystem_path(source).read_bytes() == original_bytes
    assert original_bytes.startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("raises", [False, True])
def test_image_encoding_failure_keeps_existing_file(tmp_path, monkeypatch, raises):
    target = long_unicode_root(tmp_path) / "reference.png"
    pixels = np.full((19, 23, 3), 177, np.uint8)
    write_image(target, pixels)
    original = filesystem_path(target).read_bytes()

    def failed_encode(*args, **kwargs):
        if raises:
            raise cv2.error("Controlled encoder failure")
        return False, None

    monkeypatch.setattr(cv2, "imencode", failed_encode)
    with pytest.raises(OSError, match="Could not encode image"):
        write_image(target, pixels)
    assert filesystem_path(target).read_bytes() == original


def test_unreadable_long_unicode_image_has_a_clear_error(tmp_path):
    path = long_unicode_root(tmp_path) / "empty.png"
    filesystem_path(path).write_bytes(b"")
    with pytest.raises(ValueError, match="Unreadable image"):
        read_image(path)


def test_geometry_long_unicode_roundtrip_uses_binary_handles(tmp_path, monkeypatch):
    path = long_unicode_root(tmp_path) / "模型-geometry.npz"
    profile = GeometryProfile("ribbed_face")
    profile.fit([housing(True)])
    original_save, original_load = np.savez_compressed, np.load
    calls = []

    def save(handle, *args, **kwargs):
        assert hasattr(handle, "write")
        calls.append("save")
        return original_save(handle, *args, **kwargs)

    def load(handle, *args, **kwargs):
        assert hasattr(handle, "read")
        calls.append("load")
        return original_load(handle, *args, **kwargs)

    monkeypatch.setattr(np, "savez_compressed", save)
    monkeypatch.setattr(np, "load", load)
    profile.save(path)
    loaded = GeometryProfile.load(path)

    assert calls == ["save", "load"]
    expected = profile.inspect(housing(True, missing=12))
    actual = loaded.inspect(housing(True, missing=12))
    assert actual.valid and actual.candidates == expected.candidates
    assert actual.score == expected.score


def test_failed_geometry_write_preserves_previous_profile_and_cleans_temporary(tmp_path, monkeypatch):
    path = long_unicode_root(tmp_path) / "geometry.npz"
    profile = GeometryProfile("ribbed_face")
    profile.fit([housing(True)])
    profile.save(path)
    original = filesystem_path(path).read_bytes()

    def failed_save(handle, *args, **kwargs):
        handle.write(b"partial archive")
        raise OSError("Controlled profile write failure")

    monkeypatch.setattr(np, "savez_compressed", failed_save)
    with pytest.raises(OSError, match="Controlled profile write failure"):
        profile.save(path)
    assert filesystem_path(path).read_bytes() == original
    assert not list(filesystem_path(path).parent.glob(".geometry-*.tmp"))
    assert GeometryProfile.load(path).inspect(housing(True)).valid


def test_native_evidence_long_unicode_roundtrip_and_display_paths(tmp_path, monkeypatch):
    root = long_unicode_root(tmp_path)
    monkeypatch.setattr(cv2, "imread", forbid_filename_codec)
    monkeypatch.setattr(cv2, "imwrite", forbid_filename_codec)
    session = InspectionSession("physical-长路径-17", ("inner_face",))
    session.record(ViewDecision("inner_face", "PASS", ["Software storage fixture"],
                                metrics={"source_path": filesystem_path(root / "原始图.png")}))
    session.finalize()
    frame = np.arange(73 * 101 * 3, dtype=np.uint16).reshape(73, 101, 3)
    marked = frame.copy()
    marked[3:5, 4:8] = 65535
    store = EvidenceStore(root)

    folder = store.save(session, {"inner_face": frame}, {"inner_face": marked})
    record = store.read(folder)
    history = store.history()

    assert store.save(session, {"inner_face": frame}, {"inner_face": marked}) == folder
    assert record["integrity_verified"] and len(history) == 1
    assert record["images"]["inner_face"]["original"]["shape"] == list(frame.shape)
    assert record["images"]["inner_face"]["original"]["dtype"] == "uint16"
    assert record["evidence_path"] == display_path(folder)
    assert history[0]["evidence_path"] == display_path(folder)
    assert not record["evidence_path"].startswith("\\\\?\\")
    assert record["views"]["inner_face"]["metrics"]["source_path"] == display_path(root / "原始图.png")
