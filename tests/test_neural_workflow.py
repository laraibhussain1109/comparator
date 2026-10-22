"""Opt-in actual-network integration. Synthetic housings do not validate accuracy."""
from __future__ import annotations
import os
import shutil
from pathlib import Path
import cv2
import pytest

from src.artifacts import ArtifactStore
from src.data import split_by_part
from src.utils.filesystem import filesystem_path

from test_smoke_training import make_service, import_housings
from test_vision import housing


@pytest.mark.skipif(not os.environ.get('NEUROIRIS_REAL_EFFICIENTAD_SMOKE'), reason='requires pinned neural runtime and verified pretrained assets')
def test_actual_neural_service_train_reload_tiling_calibration_evaluation(tmp_path):
    import torch
    torch.set_num_threads(2)
    base = tmp_path
    while len(str(base)) < 285:
        base /= 'NeuroIris_铝件_परीक्षण_' + str(len(base.parts))
    base = filesystem_path(base)
    base.mkdir(parents=True, exist_ok=True)
    # Exercise real teacher reads, natural-photo reads and tensor checkpoint
    # writes beyond MAX_PATH as well as the reporting/geometry path.
    teacher_cache = base / 'verified teacher cache 预训练'
    natural_folder = base / 'natural photographs 自然图像'
    teacher_cache.mkdir()
    natural_folder.mkdir()
    for name in ('pretrained_teacher_small.pth', 'pretrained_teacher_small.integrity.json'):
        shutil.copyfile(filesystem_path(Path(os.environ['NEUROIRIS_TEACHER_CACHE']) / name),
                        teacher_cache / name)
    source_photo = next(path for path in filesystem_path(os.environ['NEUROIRIS_NATURAL_IMAGE_DIR']).rglob('*')
                        if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    shutil.copyfile(source_photo, natural_folder / source_photo.name)
    service = make_service(base)
    service.config['models']['efficientad'].update({
        'max_steps': 1, 'input_size': 512, 'device': 'cpu',
        'teacher_cache': str(teacher_cache),
        'imagenet_dir': str(natural_folder),
        'allow_download': False, 'download_imagenette': False,
    })
    import_housings(service)
    split = split_by_part(service.dataset.inventory(), seed=service.config['runtime']['random_seed'],
                          ratios=tuple(service.config.get('dataset', {}).get('split', [.7, .15, .15])),
                          group_sessions=service.config.get('dataset', {}).get('group_sessions', False))
    training = [sample for sample in split.train if sample.view == 'ribbed_face' and sample.label == 0]
    blurred = service.capture_sample(cv2.GaussianBlur(housing(), (25, 25), 5),
                                     'ribbed_face', 'GOOD', training[0].part_id, 'software-fixture')
    summary = service.train('ribbed_face', 'efficientad')
    screening = summary['training_screening']
    assert screening['excluded_images'] == 1
    assert screening['accepted_images'] == summary['training_images'] == len(training)
    assert 'insufficient_focus' in screening['excluded'][0]['reasons']
    metadata = ArtifactStore.verify(Path(summary['artifact_path']))
    assert blurred['id'] not in metadata['training_accepted_sample_ids']
    assert {row['id'] for row in metadata['split']['train']} == {s.id for s in split.train} | {blurred['id']}
    assert (filesystem_path(summary['artifact_path']) / 'neural').is_dir()
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
