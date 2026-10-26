"""Reject unusable held-out captures before neural work; fixtures are synthetic."""
from pathlib import Path

import cv2
import pytest

from src.models import EfficientADBackend
from test_smoke_training import import_housings, make_service
from test_training_screening import VIEW, current_split, dataset_bytes
from src.utils.image_utils import read_image


@pytest.mark.parametrize('failure', ['blur', 'resolution'])
def test_invalid_validation_stops_before_neural_fit_and_preserves_prior_model(tmp_path, monkeypatch, failure):
    service = make_service(tmp_path)
    import_housings(service)
    service.train(VIEW, 'geometry')
    active = service.artifacts.active(VIEW, 'geometry')
    manifest = (active / 'manifest.json').read_bytes()
    original_split = current_split(service)
    reserved = [s for s in original_split.validation if s.view == VIEW and s.label == 0]
    assert reserved
    # Recapture the same reserved physical parts; never move their IDs or use
    # training captures to calibrate. The other view keeps each ID in the roster.
    for sample in reserved:
        image = read_image(sample.path)
        image = (cv2.GaussianBlur(image, (25, 25), 5) if failure == 'blur'
                 else cv2.resize(image, None, fx=.5, fy=.5))
        service.remove_sample(sample.id)
        service.capture_sample(image, VIEW, 'GOOD', sample.part_id, 'software-fixture-recapture')
    split = current_split(service)
    for partition in ('train', 'validation', 'test'):
        assert {s.part_id for s in getattr(split, partition)} == {
            s.part_id for s in getattr(original_split, partition)}
    immutable = dataset_bytes(service)
    fit_calls = []

    def forbidden_fit(*args, **kwargs):
        fit_calls.append(True)
        pytest.fail('An unusable validation set must stop before any neural fitting or asset setup')

    monkeypatch.setattr(EfficientADBackend, 'fit', forbidden_fit)
    test_paths = {s.path for s in split.test}

    def checked_read(path):
        assert Path(path) not in test_paths, 'Preflight must not inspect untouched test captures'
        return read_image(path)

    monkeypatch.setattr('src.service.read_image', checked_read)
    progress = []
    with pytest.raises(ValueError, match='No independent GOOD validation captures') as raised:
        service.train(VIEW, 'efficientad', progress=lambda *args: progress.append(args))

    assert not fit_calls
    report = service.training_screening(VIEW)
    validation = report['validation_screening']
    assert report['status'] == 'failed' and report['accepted_images'] > 0
    assert validation['phase'] == 'preflight'
    assert validation['accepted_good_images'] == 0
    assert validation['candidate_good_images'] == validation['excluded_good_images'] == len(reserved)
    assert validation['eligible_good_part_ids'] == []
    assert max(row[0] for row in progress) < 25
    rows = report['invalid_validation_captures']
    assert len(rows) == len(reserved)
    assert {row['part_id'] for row in rows} == {s.part_id for s in reserved}
    assert all(row['label'] == 'GOOD' for row in rows)
    if failure == 'blur':
        assert all(row['stage'] == 'quality' for row in rows)
        assert all('insufficient_focus' in row['reasons'] for row in rows)
        assert all(row['metrics']['sharpness'] < service.config['quality']['min_sharpness'] for row in rows)
        assert 'insufficient_focus' in str(raised.value)
    else:
        assert all(row['stage'] == 'resolution' for row in rows)
        assert all(row['metrics']['resolution'] == [288, 256] for row in rows)
        assert all(row['metrics']['expected_resolutions'] == [[576, 512]] for row in rows)
        assert 'Input resolution' in str(raised.value)
    assert report['report_path'] in str(raised.value)
    assert all(s.part_id in str(raised.value) for s in reserved)
    assert service.artifacts.active(VIEW, 'geometry') == active
    assert (active / 'manifest.json').read_bytes() == manifest
    assert dataset_bytes(service) == immutable


def test_successful_geometry_calibration_records_validation_counts_and_fixed_roster(tmp_path):
    service = make_service(tmp_path)
    import_housings(service)
    split = current_split(service)
    summary = service.train(VIEW, 'geometry')
    report = summary['training_screening']
    validation = report['validation_screening']
    expected = [s for s in split.validation if s.view == VIEW]
    assert validation['phase'] == 'calibration'
    assert validation['candidate_images'] == validation['accepted_images'] == len(expected)
    assert validation['accepted_good_images'] == len(expected)
    assert validation['excluded_images'] == validation['excluded_good_images'] == 0
    assert set(validation['accepted_sample_ids']) == {s.id for s in expected}
    assert set(validation['eligible_good_part_ids']) == {s.part_id for s in expected}
    assert report['invalid_validation_captures'] == []
