"""Versioned, integrity-checked model bundles with atomic activation."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid


def safe_name(value: str) -> str:
    if not value or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', value) or value in {'.', '..'}:
        raise ValueError('Use a name of 1–100 letters, digits, dots, underscores or hyphens')
    return value


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, allow_nan=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ArtifactStore:
    def __init__(self, root: Path, product: str):
        self.root = Path(root) / safe_name(product)

    @contextmanager
    def staging(self, view: str):
        parent = self.root / safe_name(view)
        parent.mkdir(parents=True, exist_ok=True)
        temporary = parent / ('.staging-' + uuid.uuid4().hex)
        temporary.mkdir()
        try:
            yield temporary
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def publish(self, staging: Path, metadata: dict) -> Path:
        """Only a fully written bundle becomes active; an earlier model stays intact."""
        version = safe_name(metadata.get('version') or (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]))
        metadata = {**metadata, 'schema_version': 1, 'version': version,
                    'created_at': datetime.now(timezone.utc).isoformat()}
        metadata['files'] = {str(p.relative_to(staging)): digest(p)
                             for p in sorted(staging.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
        atomic_json(staging / 'manifest.json', metadata)
        self.verify(staging)
        target = staging.parent / version
        os.replace(staging, target)
        backend = safe_name(metadata['backend'])
        atomic_json(target.parent / f'active-{backend}.json', {'version': version})
        if backend == 'efficientad':
            atomic_json(target.parent / 'active-hybrid.json', {'version': version})
        return target

    @staticmethod
    def verify(directory: Path) -> dict:
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('schema_version') != 1 or not manifest.get('files'):
            raise ValueError('Unsupported or incomplete model bundle')
        for name, checksum in manifest['files'].items():
            path = directory / name
            if not path.resolve().is_relative_to(directory.resolve()) or path.is_symlink():
                raise ValueError('Model bundle contains an unsafe path')
            if not path.is_file() or digest(path) != checksum:
                raise ValueError(f'Model bundle integrity check failed: {name}')
        return manifest

    def active(self, view: str, backend: str) -> Path:
        pointer = self.root / safe_name(view) / f'active-{safe_name(backend)}.json'
        if not pointer.is_file():
            raise FileNotFoundError(f'No trained {backend} model for {view}. Import independent physical parts and train this view first.')
        version = safe_name(json.loads(pointer.read_text(encoding='utf-8'))['version'])
        directory = pointer.parent / version
        self.verify(directory)
        return directory

    def select(self, view: str, version: str, backend: str) -> None:
        path = self.root / safe_name(view) / safe_name(version)
        metadata = self.verify(path)
        if backend in {'efficientad', 'hybrid'} and metadata['backend'] != 'efficientad':
            raise ValueError('This bundle does not contain EfficientAD weights')
        atomic_json(path.parent / f'active-{safe_name(backend)}.json', {'version': version})

    def list(self) -> list[dict]:
        result = []
        for path in sorted(self.root.glob('*/*/manifest.json'), reverse=True):
            try:
                meta = self.verify(path.parent)
                result.append({**meta, 'path': str(path.parent), 'status': 'integrity verified'})
            except (OSError, ValueError, KeyError) as exc:
                result.append({'path': str(path.parent), 'status': 'invalid', 'error': str(exc)})
        return result
