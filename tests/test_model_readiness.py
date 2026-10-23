"""Saved-model setup diagnostics are independent of inspection predictions."""
from __future__ import annotations

import pytest

from src.service import NeuroIrisService
from src.utils.filesystem import display_path
from test_smoke_training import make_service


def saved_bundle(service, view, backend='geometry'):
    # Explicit artifact-availability fixture. These bytes never run inference.
    with service.artifacts.staging(view) as directory:
        for name in ('template.png', 'geometry.npz', 'thresholds.json'):
            (directory / name).write_bytes(b'saved-model setup fixture')
        if backend == 'efficientad':
            (directory / 'neural').mkdir()
            (directory / 'neural' / 'efficientad.pt').write_bytes(b'not real neural weights')
            (directory / 'neural' / 'model.json').write_text('{}')
        return service.artifacts.publish(directory, {
            'product': service.config['product'], 'view': view, 'backend': backend,
            'version': f'fixture-{view}-{backend}', 'settings': service._settings_snapshot(),
            'rois': [],
        })


def test_no_saved_models_identifies_both_required_faces_and_exact_model_root(tmp_path):
    service = make_service(tmp_path)
    service.config['models']['backend'] = 'hybrid'

    report = service.model_readiness()

    assert report['ready'] is False
    assert report['backend'] == 'hybrid'
    assert report['artifact_root'] == display_path(service.artifacts.root)
    assert [row['view'] for row in report['views']] == ['inner_face', 'ribbed_face']
    assert all(row['status'] == 'missing' and not row['ready'] for row in report['views'])
    assert 'Face A' in report['message'] and 'Face B' in report['message']
    assert report['artifact_root'] in report['message']
    assert 'Train Model' in report['message']


def test_geometry_trained_but_hybrid_selected_reports_mode_mismatch_without_fallback(tmp_path):
    service = make_service(tmp_path)
    saved_bundle(service, 'inner_face')
    service.config['models']['backend'] = 'hybrid'

    report = service.model_readiness()

    face = report['views'][0]
    assert face['ready'] is False and face['available_modes'] == ['geometry']
    assert 'available: geometry' in report['message']
    assert service.config['models']['backend'] == 'hybrid'
    assert not (service.artifacts.root / 'inner_face' / 'active-hybrid.json').exists()


def test_face_readiness_is_independent_and_survives_restart(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    first = saved_bundle(service, 'inner_face')

    def forbidden(*args, **kwargs):
        pytest.fail('Dashboard setup checks must not load or run a model')

    monkeypatch.setattr('src.service.ViewPipeline', forbidden)
    partial = service.model_readiness()
    assert [row['ready'] for row in partial['views']] == [True, False]
    assert partial['views'][0]['path'] == display_path(first)
    saved_bundle(service, 'ribbed_face')
    restarted = NeuroIrisService(service.config)
    complete = restarted.model_readiness()
    assert complete['ready'] is True
    assert all(row['status'] == 'ready' for row in complete['views'])
    assert restarted._pipelines == {}


def test_missing_hybrid_alias_can_use_saved_neural_bundle_without_loading_torch(tmp_path):
    service = make_service(tmp_path)
    service.config['models']['backend'] = 'hybrid'
    bundle = saved_bundle(service, 'inner_face', 'efficientad')
    (bundle.parent / 'active-hybrid.json').unlink()

    report = service.model_readiness()

    assert report['views'][0]['ready'] is True
    assert report['views'][0]['version'] == bundle.name
    assert report['ready'] is False  # The opposite face still has no model.


def test_optional_enabled_face_can_be_checked_without_changing_required_coverage(tmp_path):
    service = make_service(tmp_path)
    saved_bundle(service, 'side_left')

    assert service.model_readiness()['ready'] is False
    optional = service.model_readiness(views=['side_left'])

    assert optional['ready'] is True
    assert [row['view'] for row in optional['views']] == ['side_left']
    assert service.config['views']['required'] == ['inner_face', 'ribbed_face']
    with pytest.raises(ValueError, match='enabled housing views'):
        service.model_readiness(views=['unknown'])


@pytest.mark.parametrize('change', ['settings', 'roi', 'missing_file'])
def test_unusable_saved_model_explains_setup_problem(tmp_path, change):
    service = make_service(tmp_path)
    directory = saved_bundle(service, 'inner_face')
    if change == 'settings':
        service.config['quality']['min_sharpness'] += 1
        expected = 'settings changed'
    elif change == 'roi':
        from src.artifacts import atomic_json
        atomic_json(service._roi_path('inner_face'), [{'shape': 'circle', 'points': [[.5, .5], [.1, .1]]}])
        expected = 'ROI configuration changed'
    else:
        (directory / 'template.png').unlink()
        expected = 'template.png'

    report = service.model_readiness()

    assert report['views'][0]['ready'] is False
    assert expected in report['views'][0]['reason']
    assert service._pipelines == {}


def test_legacy_joblib_models_are_identified_without_loading_or_reusing_them(tmp_path):
    service = make_service(tmp_path)
    old = service.artifacts.root.parent / 'patchcore' / 'model.joblib'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'legacy software fixture; never deserialize')

    report = service.model_readiness()

    assert report['legacy_models_found'] is True
    assert report['ready'] is False
    assert 'First-version model files' in report['message']
    assert 'view-specific model format' in report['message']


def test_missing_model_explains_the_latest_training_setup_failure(tmp_path):
    from src.artifacts import atomic_json
    service = make_service(tmp_path)
    error = 'HTTPS certificate verification failed: unable to get local issuer certificate'
    atomic_json(service._screening_directory('inner_face') / 'latest.json', {
        'status': 'failed', 'backend': 'efficientad', 'error': error,
    })

    report = service.model_readiness()

    assert report['views'][0]['ready'] is False
    assert report['views'][0]['last_training_error'] == error
    assert 'Last training did not complete' in report['views'][0]['reason']
