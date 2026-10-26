"""Capture diagnostics retain the gate and actionable failure behind RECHECK.

Software housings exercise the real quality/registration/geometry path; they
do not establish industrial accuracy or camera performance.
"""
from dataclasses import replace

import cv2
import numpy as np
import pytest

from src.acquisition import QualityChecker
from src.inspection.view_pipeline import ViewPipeline
from src.vision import GeometryProfile, HousingLocator, LandmarkRegistrar
from src.vision.geometry import GeometryResult
from test_vision import housing


@pytest.fixture
def pipeline():
    image = housing()
    registrar = LandmarkRegistrar(image)
    registration = registrar.register(image)
    assert registration.valid
    geometry = GeometryProfile('ribbed_face')
    geometry.fit([registration.image])
    # Bypass bundle I/O only. All capture gates below use their real components.
    pipeline = ViewPipeline.__new__(ViewPipeline)
    pipeline.metadata = {'source_resolutions': [[image.shape[1], image.shape[0]]],
                         'version': 'software-diagnostics-fixture'}
    pipeline.view = 'ribbed_face'
    pipeline.mode = 'geometry'
    pipeline.settings = {'registration': {'min_confidence': .6}}
    pipeline.registrar = registrar
    pipeline.geometry = geometry
    pipeline.quality = QualityChecker()
    pipeline.locator = HousingLocator()
    pipeline.rois = []
    pipeline.model = None
    return pipeline


def test_valid_capture_has_no_failure_stage(pipeline):
    data = pipeline.analyze(housing())
    assert data.valid and data.failure_stage is None and not data.reasons


def test_unrepresented_resolution_is_distinct_from_focus_or_landmark_failure(pipeline):
    frame = cv2.resize(housing(), None, fx=.5, fy=.5)
    decision, marked, data = pipeline.inspect(frame)
    assert not data.valid and data.failure_stage == 'resolution'
    assert data.quality is None and data.registration is None
    assert 'resolution' in decision.reasons[0]
    assert decision.verdict == 'RECHECK'
    assert decision.metrics['failure_stage'] == 'resolution'
    assert np.array_equal(marked, frame)


def test_blurred_capture_retains_quality_measurements(pipeline):
    frame = cv2.GaussianBlur(housing(), (25, 25), 5)
    data = pipeline.analyze(frame)
    assert not data.valid and data.failure_stage == 'quality'
    assert 'insufficient_focus' in data.reasons
    assert data.quality.metrics['sharpness'] < pipeline.quality.settings['min_sharpness']
    assert data.registration is None


@pytest.mark.parametrize('confidence', [np.nan, np.inf, -np.inf, .1])
def test_invalid_registration_confidence_cannot_pass_or_report_success(pipeline, monkeypatch,
                                                                     confidence):
    registered = pipeline.registrar.register(housing())
    assert registered.valid and registered.reasons
    monkeypatch.setattr(pipeline.registrar, 'register',
                        lambda frame: replace(registered, confidence=confidence))
    data = pipeline.analyze(housing())
    assert not data.valid and data.failure_stage == 'registration'
    assert data.geometry is None
    assert data.reasons != registered.reasons
    assert 'confidence' in data.reasons[0]


def test_missing_aligned_housing_is_distinct_from_original_capture_quality(pipeline, monkeypatch):
    original_detect = pipeline.locator.detect
    calls = 0

    def detect(frame):
        nonlocal calls
        calls += 1
        return original_detect(frame) if calls == 1 else None

    monkeypatch.setattr(pipeline.locator, 'detect', detect)
    data = pipeline.analyze(housing())
    assert not data.valid and data.failure_stage == 'aligned_pose'
    assert data.quality.valid and data.registration.valid
    assert 'Registered housing' in data.reasons[0]


def test_empty_inspection_roi_is_distinct_from_landmark_failure(pipeline):
    pipeline.rois = [{'shape': 'rectangle', 'points': [[0., 0.], [1., 1.]],
                      'exclusion': True}]
    data = pipeline.analyze(housing())
    assert not data.valid and data.failure_stage == 'roi'
    assert data.registration.valid
    assert data.reasons == ['Inspection ROI contains no part pixels']


def test_geometry_failure_preserves_specific_measurement_error(pipeline, monkeypatch):
    message = 'Housing geometry is not aligned with the GOOD profile'
    monkeypatch.setattr(pipeline.geometry, 'inspect', lambda *args, **kwargs:
                        GeometryResult(measurements={'error': message, 'center_error_px': 18.}))
    decision, _, data = pipeline.inspect(housing())
    assert not data.valid and data.failure_stage == 'geometry'
    assert data.registration.valid and data.quality.valid
    assert data.geometry.measurements['center_error_px'] == 18.
    assert decision.verdict == 'RECHECK' and decision.reasons == [message]
    assert decision.metrics['failure_stage'] == 'geometry'
