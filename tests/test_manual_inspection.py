"""Manual inspection uses genuine images and keeps operator provenance in evidence."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.service import NeuroIrisService


@pytest.fixture
def service(tmp_path, monkeypatch):
    config = yaml.safe_load((Path(__file__).parents[1] / 'config.yaml').read_text())
    config['paths'] = {name: tmp_path / name for name in config['paths']}
    instance = NeuroIrisService(config)

    def forbidden(*args, **kwargs):
        raise AssertionError('Manual inspection must not run a model')

    monkeypatch.setattr(instance, '_infer', forbidden)
    return instance


def image(value=90):
    return np.full((23, 31, 3), value, np.uint8)


def test_manual_endpoint_requires_explicit_mode_and_cannot_mix_active_sessions(service):
    with pytest.raises(RuntimeError, match='manual inspection mode'):
        service.record_manual_decision('inner_face', 'PASS', 'part-1', image())
    assert service.session is None
    service.begin_part('automatic-part')
    with pytest.raises(RuntimeError, match='current part'):
        service.set_inspection_mode('manual')
    service.cancel_part()
    service.set_inspection_mode('manual')
    assert service.begin_part('manual-part')['inspection_mode'] == 'manual'
    with pytest.raises(RuntimeError, match='current part'):
        service.set_inspection_mode('automatic')
    with pytest.raises(RuntimeError, match='automatic inspection mode'):
        service.inspect_frame(image(), 'inner_face', 'manual-part')
    with pytest.raises(RuntimeError, match='automatic inspection mode'):
        service.inspect_live('inner_face', 'manual-part')
    assert service.session.decisions == {}


@pytest.mark.parametrize('verdict', ['PASS', 'FAIL', 'RECHECK'])
def test_manual_decisions_roundtrip_once_without_models(service, verdict):
    service.set_inspection_mode('manual')
    source = image()
    result = service.record_manual_decision('inner_face', verdict, 'part-1', source)
    assert result['verdict'] == verdict
    assert result['metrics']['decision_source'] == 'operator'
    assert result['metrics']['display_decision'] == {'PASS': 'OK', 'FAIL': 'NG', 'RECHECK': 'View Recheck'}[verdict]
    assert result['model_version'] is None
    source[:] = 0
    assert np.all(service.frames['inner_face'] == 90)
    if verdict == 'PASS':
        # Required coverage still applies to a manual acceptance.
        assert result['session']['verdict'] == 'RECHECK'
        service.record_manual_decision('ribbed_face', 'PASS', 'part-1', image(100))
    final = service.finish_part()
    assert final['verdict'] == verdict
    assert service.finish_part()['evidence_path'] == final['evidence_path']
    restarted = NeuroIrisService(service.config)
    rows = restarted.history()
    assert len(rows) == 1
    row = rows[0]
    assert row['inspection_mode'] == 'manual'
    assert row['integrity_verified'] and row['counted']
    assert row['final_decision']['metrics']['decision_source'] == 'operator'
    assert row['final_decision']['metrics']['display_decision'] == result['metrics']['display_decision']
    assert row['views']['inner_face']['metrics']['inspection_scope'] == 'manual'
    assert row['views']['inner_face']['model_version'] is None
    assert 'calibrated' not in ' '.join(final['reasons'])
    assert restarted.inspection_mode == 'automatic'


def test_manual_retake_preserves_attempts_and_failure_lock(service):
    service.set_inspection_mode('manual')
    service.record_manual_decision('inner_face', 'RECHECK', 'part-1', image())
    service.record_manual_decision('inner_face', 'PASS', 'part-1', image(120))
    with pytest.raises(ValueError, match='invalid RECHECK'):
        service.record_manual_decision('inner_face', 'FAIL', 'part-1', image())
    service.record_manual_decision('ribbed_face', 'FAIL', 'part-1', image(130))
    with pytest.raises(RuntimeError, match='locked'):
        service.record_manual_decision('ribbed_face', 'PASS', 'part-1', image())
    service.finish_part()
    row = service.history()[0]
    assert [a['action'] for a in row['attempts']] == ['capture', 'retake', 'capture', 'finalize']
    assert row['attempts'][0]['decision']['verdict'] == 'RECHECK'


@pytest.mark.parametrize('view, verdict, frame', [
    ('auto', 'PASS', image()), ('missing_face', 'PASS', image()),
    ('inner_face', 'OK', image()), ('inner_face', 'PASS', None),
    ('inner_face', 'PASS', np.zeros((0, 10, 3), np.uint8)),
])
def test_invalid_manual_input_does_not_begin_a_part(service, view, verdict, frame):
    service.set_inspection_mode('manual')
    with pytest.raises(ValueError):
        service.record_manual_decision(view, verdict, 'part-1', frame)
    assert service.session is None


def test_wrong_part_and_failed_save_cannot_discard_manual_evidence(service, monkeypatch):
    service.set_inspection_mode('manual')
    service.record_manual_decision('inner_face', 'PASS', 'part-1', image())
    with pytest.raises(ValueError, match='active physical part'):
        service.record_manual_decision('ribbed_face', 'PASS', 'part-2', image())
    real_save = service.evidence.save

    def fail(*args, **kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(service.evidence, 'save', fail)
    with pytest.raises(OSError, match='disk full'):
        service.finish_part()
    with pytest.raises(RuntimeError, match='Evidence was not saved'):
        service.set_inspection_mode('automatic')
    with pytest.raises(RuntimeError, match='Evidence was not saved'):
        service.begin_part('part-2')
    monkeypatch.setattr(service.evidence, 'save', real_save)
    service.finish_part()
    service.set_inspection_mode('automatic')
