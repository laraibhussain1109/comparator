"""Landmark registration API.

Migration: replace the deprecated silhouette-only PartRegistrar with
LandmarkRegistrar(template_bgr, settings).register(frame). The Registration
result explicitly reports validity, confidence, native aligned image, and
forward/inverse coordinate transforms. Uncertain angular pose requires RECHECK.
"""
from src.vision.registration import LandmarkRegistrar, Registration

__all__ = ["LandmarkRegistrar", "Registration"]
