"""Application integration with synthetic fixtures, never industrial accuracy."""
from pathlib import Path
import json
import cv2
import numpy as np
import pytest
import yaml

from src.service import NeuroIrisService
from src.utils.config import load_config
from src.artifacts import ArtifactStore
from test_vision import housing


def make_service(tmp_path):
    config = yaml.safe_load((Path(__file__).parents[1] / 'config.yaml').read_text())
    config['models']['backend'] = 'geometry'
    config['paths'] = {key: './' + key for key in ('dataset', 'artifacts', 'results', 'logs', 'runtime')}
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    return NeuroIrisService(load_config(path))


def import_housings(service):
    for index in range(7):
        for view, ribbed in [('inner_face', False), ('ribbed_face', True)]:
            image = housing(ribbed)
            # Distinct captures of explicitly simulated physical IDs; small lighting variation.
            mask = cv2.inRange(image, (100, 100, 100), (240, 240, 240))
            image[mask > 0] = np.clip(image[mask > 0].astype(np.int16) + index - 3, 0, 255).astype(np.uint8)
            image[0, index] = [index, 0, 0]
            service.capture_sample(image, view, 'GOOD', f'synthetic-part-{index}', 'software-fixture')


def test_grouped_geometry_train_restart_inspect_and_blind_evaluate(tmp_path):
    service = make_service(tmp_path)
    import_housings(service)
    for view in ('inner_face', 'ribbed_face'):
        summary = service.train(view, 'geometry')
        assert summary['test']['count'] >= 1
        assert summary['backend'] == 'geometry'
        assert Path(summary['artifact_path']).is_dir()
        metadata = ArtifactStore.verify(Path(summary['artifact_path']))
        assert not set(metadata['training_parts']) & set(metadata['test_parts'])
        assert metadata['settings']['tiling']['tile_size'] == 512
    restarted = NeuroIrisService(service.config)
    restarted.begin_part('manual-fixture-1')
    first = restarted.inspect_frame(housing(False), 'inner_face', 'manual-fixture-1')
    assert first['session']['verdict'] == 'RECHECK'
    assert first['marked_image'].shape == housing(False).shape
    second = restarted.inspect_frame(housing(True), 'ribbed_face', 'manual-fixture-1')
    assert second['verdict'] in {'PASS', 'RECHECK'}
    final = restarted.finish_part()
    assert Path(final['evidence_path']).is_dir()
    again = restarted.finish_part()
    assert again['evidence_path'] == final['evidence_path']
    history = restarted.history()
    assert len(history) == 1 and history[0]['counted']
    report = restarted.evaluate('ribbed_face', 'geometry')
    assert report['records'] and all(row['error'] is None for row in report['records'])
    assert report['independent_ng_parts'] == 0
    assert report['parts']['recall'] is None


def test_failed_training_keeps_previous_model_and_roi_change_requires_retrain(tmp_path):
    service = make_service(tmp_path)
    import_housings(service)
    summary = service.train('ribbed_face', 'geometry')
    before = service.artifacts.active('ribbed_face', 'geometry')
    with pytest.raises((RuntimeError, ValueError, ImportError)):
        service.train('ribbed_face', 'efficientad')  # no neural dependencies/natural assets in base environment
    assert service.artifacts.active('ribbed_face', 'geometry') == before
    service.save_rois('ribbed_face', [{'shape': 'circle', 'points': [[.5, .5], [.6, .5]], 'exclusion': True}])
    outcome, _, _ = service._infer(housing(), 'ribbed_face', 'geometry')
    assert outcome.verdict == 'RECHECK' and 'ROI' in outcome.reasons[0]


def test_missing_models_missing_views_and_evidence_retry(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    service.begin_part('fixture')
    outcome = service.inspect_frame(housing(False), 'inner_face', 'fixture')
    assert outcome['verdict'] == 'RECHECK'
    assert outcome['session']['missing_views'] == ['ribbed_face']
    with pytest.raises(RuntimeError):
        service.save_rois('inner_face', [])
    original_save = service.evidence.save
    monkeypatch.setattr(service.evidence, 'save', lambda *args: (_ for _ in ()).throw(OSError('disk failure')))
    with pytest.raises(OSError):
        service.finish_part()
    with pytest.raises(RuntimeError):
        service.begin_part('next')
    monkeypatch.setattr(service.evidence, 'save', original_save)
    assert service.finish_part()['verdict'] == 'RECHECK'
    service.begin_part('next')
    with pytest.raises(RuntimeError):
        service.inspect_live('inner_face', 'next')


def test_bundle_tampering_is_not_loaded(tmp_path):
    service = make_service(tmp_path)
    import_housings(service)
    summary = service.train('inner_face', 'geometry')
    directory = Path(summary['artifact_path'])
    (directory / 'template.png').write_bytes(b'bad')
    with pytest.raises(ValueError, match='integrity'):
        ArtifactStore.verify(directory)


def test_inspection_settings_change_requires_new_calibration(tmp_path):
    service = make_service(tmp_path)
    import_housings(service)
    service.train('ribbed_face', 'geometry')
    service.configure({'geometry': {'rim_tolerance_px': 5.0}})
    outcome, _, _ = service._infer(housing(), 'ribbed_face', 'geometry')
    assert outcome.verdict == 'RECHECK' and 'settings changed' in outcome.reasons[0]
