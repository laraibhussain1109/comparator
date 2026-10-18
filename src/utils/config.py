from __future__ import annotations
import copy
import json
from pathlib import Path
from typing import Any
import yaml


def deep_merge(base: dict, update: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Path, root: Path | None = None) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle) or {}
    base = (root or path.parent).resolve()
    override = base / 'runtime' / 'config-overrides.json'
    if override.is_file():
        config = deep_merge(config, json.loads(override.read_text(encoding='utf-8')))
    config['root'] = base
    for key, value in config.get('paths', {}).items():
        candidate = Path(value)
        config['paths'][key] = (base / candidate).resolve() if not candidate.is_absolute() else candidate
    for folder in config.get('paths', {}).values():
        Path(folder).mkdir(parents=True, exist_ok=True)
    for view in config.get('views', {}).get('enabled', []):
        for label in ('GOOD', 'NG'):
            (Path(config['paths']['dataset']) / view / label).mkdir(parents=True, exist_ok=True)
    # Old regression tests use the original class-folder layout.
    for label in ('GOOD', 'NG'):
        (Path(config['paths']['dataset']) / label).mkdir(parents=True, exist_ok=True)
    return config
