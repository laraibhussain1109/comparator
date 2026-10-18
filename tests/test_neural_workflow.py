"""Opt-in actual-network integration. Synthetic housings do not validate accuracy."""
from __future__ import annotations
import os
from pathlib import Path
import pytest

from test_smoke_training import make_service, import_housings
from test_vision import housing


@pytest.mark.skipif(not os.environ.get('NEUROIRIS_REAL_EFFICIENTAD_SMOKE'), reason='requires pinned neural runtime and verified pretrained assets')
def test_actual_neural_service_train_reload_tiling_calibration_evaluation(tmp_path):
    import torch
    torch.set_num_threads(2)
    service = make_service(tmp_path)
    service.config['models']['efficientad'].update({
        'max_steps': 1, 'input_size': 512, 'device': 'cpu',
        'teacher_cache': os.environ['NEUROIRIS_TEACHER_CACHE'],
        'imagenet_dir': os.environ['NEUROIRIS_NATURAL_IMAGE_DIR'],
        'allow_download': False, 'download_imagenette': False,
    })
    import_housings(service)
    summary = service.train('ribbed_face', 'efficientad')
    assert Path(summary['artifact_path'], 'neural').is_dir()
    assert summary['test']['count'] >= 1
    assert all(row['error'] is None for row in summary['report']['records'])
    service._pipelines.clear()
    decision, marked, data = service._infer(housing(), 'ribbed_face', 'hybrid')
    assert decision.model_version == summary['version']
    assert marked.shape == housing().shape
    assert data.anomaly_map.shape == housing().shape[:2]
    assert decision.metrics['anomaly_calibrated'] is True
    report = service.benchmark('ribbed_face')
    assert report
