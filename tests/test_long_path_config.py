"""Configuration/PIN/preview I/O beyond legacy MAX_PATH, including Unicode."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.security import TechnicianLock
from src.utils.config import load_config
from src.utils.filesystem import display_path, filesystem_path


def deep_installation(tmp_path):
    directory = tmp_path.joinpath(
        *["NeuroIris aluminium housing workstation configuration segment " + str(index) for index in range(5)],
        "pièce contrôlée 铝壳",
    )
    filesystem_path(directory).mkdir(parents=True)
    assert len(str(directory)) > 260
    return directory


def configuration(**path_overrides):
    return {
        "product": "test_housing",
        "paths": {"dataset": "dataset", "artifacts": "artifacts", "results": "inspection_results",
                  "logs": "logs", "runtime": "runtime", **path_overrides},
        "views": {"enabled": ["inner_face", "ribbed_face"], "required": ["inner_face", "ribbed_face"]},
        "camera": {"index": 2, "fps": 30, "backend": "auto"},
        "quality": {"min_sharpness": 35., "min_brightness": 25.},
        # Old workflow settings must survive loading alongside modern settings.
        "image_size": 384, "patchcore": {"patch_size": 16},
    }


def write_config(directory, value):
    path = directory / "configuration 铝壳.yaml"
    filesystem_path(path).write_text(yaml.safe_dump(value, allow_unicode=True), encoding="utf-8")
    return path


def test_load_config_under_deep_unicode_installation_preserves_layouts(tmp_path):
    directory = deep_installation(tmp_path)
    loaded = load_config(write_config(directory, configuration()))
    assert display_path(loaded["root"]) == display_path(filesystem_path(directory))
    assert loaded["image_size"] == 384 and loaded["patchcore"]["patch_size"] == 16
    assert loaded["camera"] == {"index": 2, "fps": 30, "backend": "auto"}
    for name, value in loaded["paths"].items():
        assert value.is_absolute() and value.is_dir(), name
    for view in ("inner_face", "ribbed_face"):
        for label in ("GOOD", "NG"):
            assert (loaded["paths"]["dataset"] / view / label).is_dir()
    assert (loaded["paths"]["dataset"] / "GOOD").is_dir()
    assert (loaded["paths"]["dataset"] / "NG").is_dir()


def test_custom_runtime_override_reloads_and_preserves_untouched_settings(tmp_path):
    directory = deep_installation(tmp_path)
    path = write_config(directory, configuration(runtime="protected runtime 技术设置"))
    runtime = filesystem_path(directory / "protected runtime 技术设置")
    runtime.mkdir()
    (runtime / "config-overrides.json").write_text(json.dumps({
        "camera": {"fps": 15}, "quality": {"min_brightness": 40.},
        "paths": {"results": "saved evidence 检验结果"},
    }, ensure_ascii=False), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["camera"] == {"index": 2, "fps": 15, "backend": "auto"}
    assert loaded["quality"] == {"min_sharpness": 35., "min_brightness": 40.}
    assert loaded["paths"]["runtime"] == runtime
    assert loaded["paths"]["results"] == filesystem_path(directory / "saved evidence 检验结果")
    assert loaded["paths"]["results"].is_dir()


def test_legacy_runtime_override_location_is_still_read_when_custom_file_absent(tmp_path):
    directory = deep_installation(tmp_path)
    path = write_config(directory, configuration(runtime="technician runtime"))
    legacy = filesystem_path(directory / "runtime")
    legacy.mkdir()
    (legacy / "config-overrides.json").write_text('{"camera": {"fps": 12}}', encoding="utf-8")
    assert load_config(path)["camera"]["fps"] == 12


def test_relative_config_and_explicit_root_keep_their_existing_meaning(tmp_path, monkeypatch):
    directory = deep_installation(tmp_path)
    path = write_config(directory, configuration())
    monkeypatch.chdir(filesystem_path(directory))
    loaded = load_config(Path(path.name), root=Path("separate runtime root 根目录"))
    expected = filesystem_path(directory / "separate runtime root 根目录")
    assert loaded["root"] == expected
    assert loaded["paths"]["dataset"] == expected / "dataset"
    assert loaded["paths"]["runtime"].is_dir()


def test_absolute_dataset_path_remains_outside_the_installation_root(tmp_path):
    directory = deep_installation(tmp_path)
    external = tmp_path / "shared housing data 外部数据"
    path = write_config(directory, configuration(dataset=str(external)))
    loaded = load_config(path)
    assert loaded["paths"]["dataset"] == filesystem_path(external)
    assert (loaded["paths"]["dataset"] / "inner_face" / "GOOD").is_dir()


def test_technician_pin_setup_and_restart_under_deep_unicode_runtime(tmp_path):
    directory = deep_installation(tmp_path)
    path = directory / "protected runtime 技术设置" / "technician-lock.json"
    pin = "local-software-fixture-pin"
    lock = TechnicianLock(path)
    assert lock.path == filesystem_path(path)
    assert not lock.configured
    lock.setup(pin)
    assert lock.configured and lock.unlocked
    raw = filesystem_path(path).read_text(encoding="utf-8")
    assert pin not in raw and "hash" in json.loads(raw)
    restarted = TechnicianLock(path)
    assert restarted.configured and not restarted.unlocked
    with pytest.raises(PermissionError):
        restarted.require()
    assert not restarted.unlock("incorrect-software-fixture-pin")
    assert restarted.unlock(pin)
    restarted.require()
    restarted.lock()
    assert not restarted.unlocked


def test_service_saved_custom_runtime_settings_reload_under_deep_installation(tmp_path):
    from src.service import NeuroIrisService

    directory = deep_installation(tmp_path)
    template = filesystem_path(Path(__file__).parents[1] / "config.yaml")
    settings = yaml.safe_load(template.read_text(encoding="utf-8"))
    settings["paths"] = configuration(runtime="protected runtime 技术设置")["paths"]
    path = write_config(directory, settings)
    service = NeuroIrisService(load_config(path))
    service.configure({"quality": {"min_sharpness": 42.}, "camera": {"fps": 15}})
    saved = service.config["paths"]["runtime"] / "config-overrides.json"
    assert saved.is_file()
    reloaded = load_config(path)
    assert reloaded["quality"]["min_sharpness"] == 42.
    assert reloaded["camera"]["fps"] == 15
    assert reloaded["camera"]["backend"] == settings["camera"]["backend"]


def test_dialog_preview_reads_long_unicode_images_without_native_filename_reader(tmp_path, monkeypatch):
    from src.ui.dialogs import read_image

    directory = deep_installation(tmp_path)
    image = np.zeros((23, 37, 3), np.uint8)
    image[4:12, 8:19] = [90, 140, 210]
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path = directory / "housing preview 检查图像.png"
    filesystem_path(path).write_bytes(encoded.tobytes())

    def forbidden(*args, **kwargs):
        raise AssertionError("Preview must use Python's normalized filesystem boundary")

    monkeypatch.setattr(np, "fromfile", forbidden)
    assert np.array_equal(read_image(str(path)), image)
    assert read_image(str(directory / "missing.png")) is None
    invalid = directory / "empty.png"
    filesystem_path(invalid).write_bytes(b"")
    assert read_image(str(invalid)) is None
