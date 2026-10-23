"""Controlled software fixtures; these are not industrial accuracy evidence."""
from __future__ import annotations

import cv2
import numpy as np

from src.vision import (FaceIdentifier, GeometryProfile, HousingLocator, LandmarkRegistrar,
                        merge_tile_maps, roi_mask, tile_image)


def housing(ribbed=True, missing=None):
    image = np.full((512, 576, 3), (155, 95, 25), np.uint8)
    center, radius = (288, 256), 190
    cv2.circle(image, center, radius, (190, 190, 190), -1)
    cv2.circle(image, center, radius - 12, (135, 135, 135), -1)
    if ribbed:
        for index in range(36):
            if index in (3, 4, missing):  # Legitimate design gaps are part of GOOD.
                continue
            angle = index * 2 * np.pi / 36
            a = tuple(np.rint(np.array(center) + .34 * radius * np.array([np.cos(angle), np.sin(angle)])).astype(int))
            b = tuple(np.rint(np.array(center) + .90 * radius * np.array([np.cos(angle), np.sin(angle)])).astype(int))
            cv2.line(image, a, b, (75, 75, 75), 7)
            cv2.line(image, (a[0] - 1, a[1] - 1), (b[0] - 1, b[1] - 1), (225, 225, 225), 3)
        cv2.rectangle(image, (259, 227), (317, 285), (165, 165, 165), -1)
        cv2.rectangle(image, (259, 227), (317, 285), (65, 65, 65), 3)
    else:
        cv2.circle(image, center, 155, (220, 220, 220), 5)
        cv2.circle(image, center, 146, (175, 175, 175), 3)
    # Distinct, asymmetric physical fixture landmarks establish angle.
    for x, y, r in [(270, 225, 6), (308, 245, 8), (273, 278, 5), (238, 285, 9), (320, 300, 4)]:
        cv2.circle(image, (x, y), r + 2, (210, 210, 210), -1)
        cv2.circle(image, (x, y), r, (30, 30, 30), -1)
    return image


def test_housing_locator_and_incomplete_capture():
    image = housing()
    pose = HousingLocator().detect(image)
    assert pose is not None and pose.complete
    assert np.linalg.norm(np.array(pose.center) - [288, 256]) < 2
    assert abs(pose.radius - 190) < 2
    assert pose.mask.shape == image.shape[:2]
    assert HousingLocator().detect(np.zeros_like(image)) is None
    clipped = image[:, 130:400]
    pose = HousingLocator().detect(clipped)
    assert pose is None or not pose.complete


def test_face_cues_are_conservative():
    identifier = FaceIdentifier()
    assert identifier.identify(housing())[0] == "ribbed_face"
    assert identifier.identify(housing(False))[0] == "inner_face"
    plain = np.zeros((400, 400, 3), np.uint8)
    cv2.circle(plain, (200, 200), 130, (180, 180, 180), -1)
    assert identifier.identify(plain)[0] is None


def test_landmark_rotation_mapping_and_circle_ambiguity():
    reference = housing()
    transform = cv2.getRotationMatrix2D((288, 256), 27, 1)
    transform[:, 2] += [7, -4]
    moving = cv2.warpAffine(reference, transform, (576, 512), borderValue=(155, 95, 25))
    registered = LandmarkRegistrar(reference).register(moving)
    assert registered.valid, registered.reasons
    points = np.float32([[[270, 225]], [[308, 245]], [[238, 285]]])
    source = cv2.transform(points, transform).reshape(-1, 2)
    mapped = cv2.perspectiveTransform(source[:, None], registered.matrix).reshape(-1, 2)
    assert np.max(np.linalg.norm(mapped - points.reshape(-1, 2), axis=1)) < 2
    assert np.max(np.abs(registered.source_points(mapped) - source)) < 1e-3
    assert registered.image.shape == reference.shape
    assert HousingLocator().detect(registered.image).complete
    plain = np.zeros_like(reference)
    cv2.circle(plain, (288, 256), 190, (180, 180, 180), -1)
    assert not LandmarkRegistrar(plain).register(plain).valid


def test_registration_exposure_change_and_repeated_rib_ambiguity():
    reference = housing()
    transform = cv2.getRotationMatrix2D((288, 256), -13, 1)
    moving = cv2.warpAffine(reference, transform, (576, 512), borderValue=(155, 95, 25))
    moving = np.clip(moving.astype(np.int16) + 11, 0, 255).astype(np.uint8)
    result = LandmarkRegistrar(reference).register(moving)
    assert result.valid, result.reasons
    point = np.float32([[[238, 285]]])
    captured = cv2.transform(point, transform)
    recovered = cv2.perspectiveTransform(captured, result.matrix)
    assert np.linalg.norm(recovered - point) < 2
    symmetric = np.full((512, 512, 3), (155, 95, 25), np.uint8)
    cv2.circle(symmetric, (256, 256), 190, (135, 135, 135), -1)
    for index in range(36):
        angle = index * np.pi / 18
        start = tuple(np.rint([256 + 60 * np.cos(angle), 256 + 60 * np.sin(angle)]).astype(int))
        end = tuple(np.rint([256 + 175 * np.cos(angle), 256 + 175 * np.sin(angle)]).astype(int))
        cv2.line(symmetric, start, end, (225, 225, 225), 4)
    rotated = cv2.warpAffine(symmetric, cv2.getRotationMatrix2D((256, 256), 20, 1),
                            (512, 512), borderValue=(155, 95, 25))
    assert not LandmarkRegistrar(symmetric).register(rotated).valid


def test_inner_face_hole_constellation_registers_and_symmetric_holes_recheck():
    reference = housing(False)
    transform = cv2.getRotationMatrix2D((288, 256), 27, 1)
    moving = cv2.warpAffine(reference, transform, (576, 512), borderValue=(155, 95, 25))
    result = LandmarkRegistrar(reference).register(moving)
    assert result.valid and "hole landmarks" in result.reasons[0]
    assert np.max(np.abs(result.matrix[:2] - cv2.invertAffineTransform(transform))) < 1
    symmetric = np.full((512, 512, 3), (155, 95, 25), np.uint8)
    cv2.circle(symmetric, (256, 256), 190, (180, 180, 180), -1)
    for index in range(6):
        angle = index * np.pi / 3
        point = (round(256 + 100 * np.cos(angle)), round(256 + 100 * np.sin(angle)))
        cv2.circle(symmetric, point, 8, (30, 30, 30), -1)
    rotated = cv2.warpAffine(symmetric, cv2.getRotationMatrix2D((256, 256), 35, 1),
                            (512, 512), borderValue=(155, 95, 25))
    assert not LandmarkRegistrar(symmetric).register(rotated).valid


def test_missing_rib_is_learned_localized_and_persistent(tmp_path):
    good = housing()
    profile = GeometryProfile("ribbed_face")
    profile.fit([good, np.clip(good.astype(np.int16) + 4, 0, 255).astype(np.uint8)])
    normal = profile.inspect(good)
    assert normal.valid
    assert not normal.candidates, normal.candidates
    defective = housing(missing=12)
    result = profile.inspect(defective)
    rib_candidates = [c for c in result.candidates if "rib" in c["kind"]]
    assert rib_candidates, result.measurements
    assert result.score > normal.score
    for candidate in rib_candidates:
        x, y, w, h = candidate["bbox"]
        assert w * h < .2 * np.pi * 190 ** 2
        # Missing rib 12 lies below and to the left of the centre.
        assert x < 288 and y > 270
        assert candidate["contour"]
    path = tmp_path / "geometry.npz"
    profile.save(path)
    loaded = GeometryProfile.load(path)
    assert loaded.inspect(defective).candidates == result.candidates


def test_exclusions_suppress_expected_keepout_without_poisoning_profile():
    good, defective = housing(), housing(missing=12)
    exclusion = np.zeros(good.shape[:2], np.uint8)
    cv2.rectangle(exclusion, (170, 300), (270, 445), 255, -1)
    profile = GeometryProfile("ribbed_face")
    profile.fit([good], exclusions=exclusion)
    assert not profile.inspect(defective).candidates
    profile = GeometryProfile("ribbed_face")
    profile.fit([good])
    assert not profile.inspect(defective, exclusion).candidates


def test_bent_rib_has_measured_displaced_edges():
    profile = GeometryProfile("ribbed_face")
    profile.fit([housing()])
    bent = housing(missing=12)
    points = []
    for radius, degrees in [(65, 120), (110, 122), (170, 124)]:
        angle = np.deg2rad(degrees)
        points.append([round(288 + radius * np.cos(angle)), round(256 + radius * np.sin(angle))])
    cv2.polylines(bent, [np.array(points)], False, (75, 75, 75), 7)
    cv2.polylines(bent, [np.array(points) - 1], False, (225, 225, 225), 3)
    result = profile.inspect(bent)
    candidates = [c for c in result.candidates if c["kind"] == "displaced_or_bent_rib"]
    assert candidates
    assert all(c["measurements"]["displacement_edge_support_ratio"] > .35 for c in candidates)


def test_polar_coordinate_roundtrip_and_localized_rim_chip():
    good = housing()
    profile = GeometryProfile("ribbed_face")
    profile.fit([good])
    marker = np.zeros(good.shape[:2], np.uint8)
    for point in [(438, 256), (288, 406), (138, 256), (288, 106)]:
        cv2.circle(marker, point, 4, 255, -1)
    recovered = profile._cartesian(profile._polar(marker, nearest=True))
    for x, y in [(438, 256), (288, 406), (138, 256), (288, 106)]:
        assert np.any(recovered[y - 2:y + 3, x - 2:x + 3])
    chipped = good.copy()
    cv2.rectangle(chipped, (463, 240), (485, 272), (155, 95, 25), -1)
    candidates = [c for c in profile.inspect(chipped).candidates if c["kind"] == "rim_shape_deviation"]
    assert candidates
    assert all(c["bbox"][0] > 440 and c["bbox"][2] < 45 for c in candidates)
    excluded = np.zeros(good.shape[:2], np.uint8)
    excluded[:, 440:] = 255
    assert not any(c["kind"] == "rim_shape_deviation" for c in profile.inspect(chipped, excluded).candidates)


def test_holes_and_inner_circle_measurements():
    good = housing(False)
    profile = GeometryProfile("inner_face")
    profile.fit([good])
    result = profile.inspect(good)
    assert result.valid and not result.candidates
    assert result.measurements["expected_repeatable_holes"] >= 3
    assert result.measurements["inner_circle"]["detected"]
    defective = good.copy()
    cv2.circle(defective, (238, 285), 12, (135, 135, 135), -1)
    assert any(c["kind"] == "missing_hole" for c in profile.inspect(defective).candidates)


def test_roi_serialization_schema_and_exclusion_precedence():
    definitions = [
        {"shape": "rectangle", "points": [[.1, .1], [.9, .9]], "exclusion": False},
        {"shape": "circle", "points": [[.5, .5], [.6, .5]], "exclusion": True},
        {"shape": "polygon", "points": [[.8, .1], [.9, .1], [.9, .2]], "exclusion": False},
    ]
    mask = roi_mask((200, 400, 3), definitions)
    assert mask[100, 200] == 0
    assert mask[40, 60] == 255
    assert mask[0, 0] == 0
    assert roi_mask((200, 400), [definitions[1]])[0, 0] == 255


def test_tiling_preserves_native_coordinates_and_overlap():
    height, width = 601, 973
    y, x = np.mgrid[:height, :width]
    image = np.dstack([x % 255, y % 255, np.zeros_like(x)]).astype(np.uint8)
    tiles = list(tile_image(image, 256, .25))
    maps = [x[tile.y:tile.y + tile.height, tile.x:tile.x + tile.width].astype(np.float32) for tile in tiles]
    merged = merge_tile_maps(tiles, maps, image.shape)
    assert merged.shape == (height, width)
    assert np.allclose(merged, x, atol=1e-3)
    assert tiles[-1].x + tiles[-1].width == width
    assert tiles[-1].y + tiles[-1].height == height
    small = list(tile_image(image[:63, :79], 256))
    assert len(small) == 1 and small[0].image.shape[:2] == (63, 79)
    mask = np.zeros((height, width), np.uint8)
    mask[-10:, -10:] = 255
    selected = list(tile_image(image, 256, .25, mask))
    assert selected and len(selected) < len(tiles)
