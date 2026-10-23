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

from src.utils.filesystem import display_path, filesystem_path


def safe_name(value: str) -> str:
    if not value or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', value) or value in {'.', '..'}:
        raise ValueError('Use a name of 1–100 letters, digits, dots, underscores or hyphens')
    return value


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with filesystem_path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path = filesystem_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Independent of the destination's basename: appending two UUIDs pushed the
    # reported Windows staging path to MAX_PATH even though the final file fit.
    temporary = path.with_name(f'.{uuid.uuid4().hex}.tmp')
    created = False
    try:
        with temporary.open('x', encoding='utf-8') as handle:
            created = True
            json.dump(value, handle, indent=2, allow_nan=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


class ArtifactStore:
    def __init__(self, root: Path, product: str):
        self.product = safe_name(product)
        self.root = filesystem_path(root) / self.product

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
        staging = filesystem_path(staging)
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
    def verify(directory: Path, *, verify_files: bool = True) -> dict:
        """Validate a bundle, optionally deferring expensive file digest checks."""
        directory = filesystem_path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if (not isinstance(manifest, dict) or manifest.get('schema_version') != 1
                or not isinstance(manifest.get('files'), dict) or not manifest['files']):
            raise ValueError('Unsupported or incomplete model bundle')
        for name, checksum in manifest['files'].items():
            if (not isinstance(name, str) or not isinstance(checksum, str)
                    or not re.fullmatch(r'[0-9a-f]{64}', checksum)):
                raise ValueError('Model bundle contains an invalid file entry')
            path = directory / name
            if not path.resolve().is_relative_to(directory.resolve()) or path.is_symlink():
                raise ValueError('Model bundle contains an unsafe path')
            if not path.is_file():
                raise ValueError(f'Model bundle file is missing: {name}')
            if verify_files and digest(path) != checksum:
                raise ValueError(f'Model bundle integrity check failed: {name}')
        return manifest

    def _validate_selection(self, directory: Path, view: str, backend: str,
                            version: str, *, verify_files: bool = True) -> dict:
        parent = self.root / safe_name(view)
        if not directory.resolve().is_relative_to(parent.resolve()):
            raise ValueError('Selected model bundle is outside its product/view directory')
        metadata = self.verify(directory, verify_files=verify_files)
        if metadata.get('product') != self.product:
            raise ValueError('Model bundle product does not match the selected product')
        if metadata.get('view') != view:
            raise ValueError('Model bundle view does not match the selected view')
        if metadata.get('version') != version:
            raise ValueError('Model bundle version does not match the model selection')
        if backend in {'efficientad', 'hybrid'} and metadata.get('backend') != 'efficientad':
            raise ValueError('This bundle does not contain EfficientAD weights')
        if backend == 'geometry' and metadata.get('backend') not in {'geometry', 'efficientad'}:
            raise ValueError('This bundle does not contain a supported geometry model')
        return metadata

    def active(self, view: str, backend: str, *, verify_files: bool = True) -> Path:
        """Resolve an explicit selection, or its compatible neural-mode alias.

        Hybrid and EfficientAD share a trained bundle. An interrupted second
        pointer write must not hide that bundle, but a present selection is
        authoritative even when corrupt. No other mode or saved version is
        selected automatically. ``verify_files=False`` supports cheap readiness
        checks; inference keeps the default digest verification enabled.
        """
        view, backend = safe_name(view), safe_name(backend)
        if backend not in {'geometry', 'efficientad', 'hybrid'}:
            raise ValueError('Select geometry, efficientad or hybrid')
        pointer = self.root / view / f'active-{backend}.json'
        if not pointer.exists() and not pointer.is_symlink():
            sibling = {'hybrid': 'efficientad', 'efficientad': 'hybrid'}.get(backend)
            alias = pointer.with_name(f'active-{sibling}.json') if sibling else None
            if alias is None or (not alias.exists() and not alias.is_symlink()):
                mode_guidance = (' Geometry-only models require Geometry mode.'
                                 if backend in {'hybrid', 'efficientad'} else '')
                raise FileNotFoundError(
                    f'No trained {backend} model for {view}. No active selection was found at '
                    f'{display_path(pointer)}. Open Model Settings to select a compatible saved '
                    f'model for this face, or complete training for this view.{mode_guidance}')
            pointer = alias
        selection = json.loads(pointer.read_text(encoding='utf-8'))
        if not isinstance(selection, dict) or not isinstance(selection.get('version'), str):
            raise ValueError(f'Invalid model selection file: {display_path(pointer)}')
        version = safe_name(selection['version'])
        directory = pointer.parent / version
        self._validate_selection(directory, view, backend, version, verify_files=verify_files)
        return directory

    def select(self, view: str, version: str, backend: str) -> None:
        view, version, backend = safe_name(view), safe_name(version), safe_name(backend)
        if backend not in {'geometry', 'efficientad', 'hybrid'}:
            raise ValueError('Select geometry, efficientad or hybrid')
        path = self.root / view / version
        self._validate_selection(path, view, backend, version)
        atomic_json(path.parent / f'active-{backend}.json', {'version': version})

    def list(self) -> list[dict]:
        result = []
        for path in sorted(self.root.glob('*/*/manifest.json'), reverse=True):
            try:
                meta = self.verify(path.parent)
                result.append({**meta, 'path': display_path(path.parent), 'status': 'integrity verified'})
            except (OSError, ValueError, KeyError) as exc:
                result.append({'path': display_path(path.parent), 'status': 'invalid', 'error': str(exc)})
        return result
