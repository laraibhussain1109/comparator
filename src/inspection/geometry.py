"""Learned native-resolution housing geometry API.

Migration: replace extract_geometry/GeometryInspector with
GeometryProfile(view, settings).fit(registered_good_images), then inspect(image).
GeometryResult carries localized candidates, measurements, and validity.
Whole-part feature deviations no longer create invented defect contours.
"""
from src.vision.geometry import GeometryProfile, GeometryResult

__all__ = ["GeometryProfile", "GeometryResult"]
