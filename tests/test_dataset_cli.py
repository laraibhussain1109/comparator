"""Explicit selected-identity repair through the operator's CLI entry point."""
import json
from pathlib import Path

from src.cli import main
from test_smoke_training import make_service
from test_vision import housing


def test_cli_repairs_empty_legacy_metadata_and_keeps_image_bytes(tmp_path, capsys):
    service = make_service(tmp_path)
    old = service.capture_sample(housing(), 'ribbed_face', 'GOOD', 'LEMON', 'oct')
    original = Path(old['path'])
    image_bytes = original.read_bytes()
    sidecar = original.with_suffix(original.suffix + '.json')
    service.dataset._mirror(original).unlink()
    sidecar.write_bytes(b'')

    result = main(['--config', str(tmp_path / 'config.yaml'), 'repair', str(original),
                   '--view', 'ribbed_face', '--label', 'GOOD', '--part-id', 'LEMON',
                   '--session-id', 'oct'])

    assert result == 0
    repaired = json.loads(capsys.readouterr().out)
    image = Path(repaired['path'])
    assert image.read_bytes() == image_bytes
    assert repaired['part_id'] == 'LEMON' and repaired['view'] == 'ribbed_face'
    assert repaired['metadata']['session_id'] == 'oct'
    metadata = json.loads(image.with_suffix(image.suffix + '.json').read_text(encoding='utf-8'))
    assert metadata == repaired['metadata']
    backup = service.dataset.root / metadata['metadata_repair_backup']
    assert backup.read_bytes() == b''
    assert not service.pending_samples()
    assert len(service.samples()) == 1
