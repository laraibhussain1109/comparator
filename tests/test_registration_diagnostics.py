"""Measured transform diagnostics do not change landmark rejection limits."""
import numpy as np
import pytest

from src.vision import LandmarkRegistrar, Registration
from test_vision import housing


def install_landmark_matches(registrar, monkeypatch, scale, translation):
    # Eight distinct mutual descriptor matches establish an exact known affine.
    # The controlled mismatch is between these landmarks and the real silhouette.
    offsets = np.float32([[-85, -70], [0, -80], [80, -65], [90, 0],
                          [70, 75], [0, 90], [-75, 65], [-90, 0]])
    center = np.array(registrar.template_pose.center, np.float32)
    points = offsets + center
    target = offsets * scale + center + np.array(translation, np.float32)
    descriptors = np.random.default_rng(17).integers(0, 256, (len(points), 32), dtype=np.uint8)
    registrar._template_points = target
    registrar._template_descriptors = descriptors
    monkeypatch.setattr(registrar, '_features', lambda image: (points, descriptors))


@pytest.mark.parametrize('scale,translation,expected_checks', [
    (1., (20., 0.), ['center']),
    (1.12, (0., 0.), ['radius']),
])
def test_disagreeing_transform_keeps_rejection_and_reports_measured_values(monkeypatch, scale,
                                                                          translation, expected_checks):
    image = housing(False)
    registrar = LandmarkRegistrar(image)
    install_landmark_matches(registrar, monkeypatch, scale, translation)
    monkeypatch.setattr(registrar, '_hole_registration', lambda image: None)

    result = registrar.register(image)

    assert not result.valid
    assert result.reasons == ['Estimated landmark transform disagrees with housing geometry']
    measured = result.metrics
    assert measured['failed_checks'] == expected_checks
    assert measured['estimated_scale'] == pytest.approx(scale, abs=1e-5)
    assert measured['center_error_px'] == pytest.approx(np.linalg.norm(translation), abs=1e-4)
    assert measured['center_error_radius'] == pytest.approx(
        np.linalg.norm(translation) / registrar.template_pose.radius, abs=1e-5)
    assert measured['radius_error_fraction'] == pytest.approx(abs(scale - 1), abs=1e-5)
    assert measured['minimum_scale'] == .5 and measured['maximum_scale'] == 2.
    assert measured['maximum_center_error_radius'] == .06
    assert measured['maximum_radius_error_fraction'] == .08
    assert measured['maximum_reprojection_error_px'] == pytest.approx(
        max(2., .008 * registrar.template_pose.radius))
    assert measured['reprojection_error_px'] < 1e-4
    assert measured['matched_landmarks'] == measured['inlier_landmarks'] == 8
    assert measured['inlier_ratio'] == 1.


def test_successful_hole_fallback_does_not_inherit_failed_orb_transform_metrics(monkeypatch):
    image = housing(False)
    registrar = LandmarkRegistrar(image)
    install_landmark_matches(registrar, monkeypatch, 1., (20., 0.))
    identity = np.eye(3)
    fallback = Registration(image.copy(), .9, identity, identity,
                            ['Controlled valid asymmetric hole constellation'], True)
    monkeypatch.setattr(registrar, '_hole_registration', lambda image: fallback)

    result = registrar.register(image)

    assert result is fallback and result.valid
    assert result.metrics == {}
    assert result.reasons == ['Controlled valid asymmetric hole constellation']
