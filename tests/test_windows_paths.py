"""Windows namespace/legacy-limit regressions, with explicit host emulation.

The MAX_PATH guard models Windows pathname handling on any test host. Native
Windows CI must independently run the workflow tests; emulation is not hardware
or operating-system acceptance evidence.
"""
import json
import os
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

import src.artifacts as artifacts
import src.utils.filesystem as filesystem


SCREENSHOT_DIRECTORY = (
    r"C:\Users\user\Downloads\comparator-codex-neuroiris-inspection-system (3)"
    r"\comparator-codex-neuroiris-inspection-system\inspection_results"
    r"\training_screening\aluminium_housing\inner_face"
)
SCREENSHOT_FINAL = "266aabe407bd4612ac3234d764a573af.json"
SCREENSHOT_OLD_TEMP = (
    ".266aabe407bd4612ac3234d764a573af.json.7b6d26841d0849438569b4291642b90f.tmp"
)


@pytest.mark.parametrize("source, expected", [
    (r"C:\inspection\铝件 परीक्षण\latest.json", r"\\?\C:\inspection\铝件 परीक्षण\latest.json"),
    (r"C:\inspection\stage\..\latest.json", r"\\?\C:\inspection\latest.json"),
    (r"\\inspection-server\quality records\铝件\latest.json",
     r"\\?\UNC\inspection-server\quality records\铝件\latest.json"),
    (r"\\?\C:\inspection\latest.json", r"\\?\C:\inspection\latest.json"),
    (r"\\?\UNC\inspection-server\share\latest.json",
     r"\\?\UNC\inspection-server\share\latest.json"),
])
def test_windows_path_namespace_normalization_is_absolute_and_idempotent(monkeypatch, source, expected):
    monkeypatch.setattr(filesystem, "WINDOWS", True)
    normalized = filesystem.filesystem_path(source)
    assert isinstance(normalized, Path)
    assert str(normalized) == expected
    assert str(filesystem.filesystem_path(normalized)) == expected


def test_non_windows_paths_keep_normal_path_semantics(tmp_path, monkeypatch):
    monkeypatch.setattr(filesystem, "WINDOWS", False)
    path = tmp_path / "原始_परीक्षण" / "latest.json"
    assert filesystem.filesystem_path(path) == path


@pytest.mark.parametrize("source, expected", [
    (r"\\?\C:\inspection\latest.json", r"C:\inspection\latest.json"),
    (r"\\?\UNC\inspection-server\share\latest.json", r"\\inspection-server\share\latest.json"),
])
def test_operator_display_omits_internal_windows_namespace(source, expected):
    assert filesystem.display_path(source) == expected


def test_exact_screenshot_staging_path_reproduces_legacy_limit_and_atomic_json_recovers(tmp_path, monkeypatch):
    windows_directory = PureWindowsPath(SCREENSHOT_DIRECTORY)
    legacy_display = str(windows_directory / SCREENSHOT_OLD_TEMP)
    final_display = str(windows_directory / SCREENSHOT_FINAL)
    assert len(final_display) <= 259 < len(legacy_display)
    host = tmp_path / "emulated_windows_directory"
    host.mkdir()
    target = host / SCREENSHOT_FINAL
    concrete_path = type(Path())

    class ExtendedHostPath(concrete_path):
        """Use real host IO while exposing the Windows extended pathname."""
        def __str__(self):
            relative = concrete_path(super().__str__()).relative_to(host)
            return "\\\\?\\" + str(windows_directory.joinpath(*relative.parts))

        def __fspath__(self):
            return super().__str__()

    def windows_display(path):
        if isinstance(path, ExtendedHostPath):
            return str(path)
        try:
            relative = concrete_path(os.fspath(path)).relative_to(host)
        except ValueError:
            return str(path)
        return str(windows_directory.joinpath(*relative.parts))

    original_open = concrete_path.open
    opened = []
    def guarded_open(path, *args, **kwargs):
        displayed = windows_display(path)
        if len(displayed) > 259 and not displayed.startswith("\\\\?\\"):
            raise FileNotFoundError(2, "Emulated legacy Windows MAX_PATH limit", displayed)
        mode = args[0] if args else kwargs.get("mode", "r")
        opened.append((path.name, mode, displayed))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(concrete_path, "open", guarded_open)
    with pytest.raises(FileNotFoundError, match="MAX_PATH"):
        (host / SCREENSHOT_OLD_TEMP).open("x", encoding="utf-8")

    normalizations = []
    def normalize_for_emulated_io(path):
        normalizations.append(os.fspath(path))
        return ExtendedHostPath(path)
    monkeypatch.setattr(artifacts, "filesystem_path", normalize_for_emulated_io)

    artifacts.atomic_json(target, {"status": "screening", "view": "inner_face"})

    assert normalizations
    assert json.loads(target.read_text(encoding="utf-8"))["status"] == "screening"
    staging = [(name, displayed) for name, mode, displayed in opened if "x" in mode or "w" in mode]
    assert staging and all(len(name) <= 48 for name, _ in staging)
    assert all(displayed.startswith("\\\\?\\") for _, displayed in staging)
    assert set(host.iterdir()) == {target}


def test_atomic_json_creates_missing_parents_and_uses_short_temporary_names(tmp_path, monkeypatch):
    target = tmp_path / "uncreated" / "nested" / ("training_screening_" + "a" * 90 + ".json")
    original_open = type(target).open
    staging = []
    def observe_open(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if "x" in mode or "w" in mode:
            staging.append(path.name)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(type(target), "open", observe_open)
    artifacts.atomic_json(target, {"message": "铝件 परीक्षा", "count": 2})
    usable_target = filesystem.filesystem_path(target)
    assert json.loads(usable_target.read_text(encoding="utf-8"))["count"] == 2
    assert staging and all(len(name) <= 48 and target.name not in name for name in staging)
    assert list(usable_target.parent.iterdir()) == [usable_target]


@pytest.mark.parametrize("failure", ["serialization", "replacement"])
def test_atomic_json_failures_preserve_previous_target_and_clean_only_new_staging(tmp_path, monkeypatch, failure):
    target = tmp_path / "saved-model-selection.json"
    old_bytes = b'{"version":"previous-valid-version"}'
    target.write_bytes(old_bytes)
    payload = {"value": float("nan")} if failure == "serialization" else {"version": "candidate"}
    if failure == "replacement":
        def reject_replace(*args, **kwargs):
            raise OSError("Controlled replacement failure")
        monkeypatch.setattr(artifacts.os, "replace", reject_replace)
    with pytest.raises((ValueError, OSError)):
        artifacts.atomic_json(target, payload)
    assert target.read_bytes() == old_bytes
    assert set(tmp_path.iterdir()) == {target}


def test_atomic_json_does_not_remove_a_preexisting_staging_file_on_collision(tmp_path, monkeypatch):
    target = tmp_path / "saved-settings.json"
    target.write_bytes(b'{"version":"previous"}')
    fixed_id = "a" * 32
    existing_temporary = tmp_path / f".{fixed_id}.tmp"
    existing_temporary.write_bytes(b"preexisting user file")
    monkeypatch.setattr(artifacts.uuid, "uuid4", lambda: SimpleNamespace(hex=fixed_id))
    with pytest.raises(FileExistsError):
        artifacts.atomic_json(target, {"version": "candidate"})
    assert target.read_bytes() == b'{"version":"previous"}'
    assert existing_temporary.read_bytes() == b"preexisting user file"
