from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


def load_config(path: Path, root: Path | None = None) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    base = (root or path.parent).resolve()
    config["root"] = base
    for key, value in config.get("paths", {}).items():
        candidate = Path(value)
        config["paths"][key] = (base / candidate).resolve() if not candidate.is_absolute() else candidate
    for folder in config.get("paths", {}).values():
        Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(config["paths"]["dataset"]) / "GOOD").mkdir(parents=True, exist_ok=True)
    (Path(config["paths"]["dataset"]) / "NG").mkdir(parents=True, exist_ok=True)
    return config
