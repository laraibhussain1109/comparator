"""Validation-only operating points constrained by GOOD-part false rejects.

Migration: calibrate_operating_point requires independent GOOD physical part IDs;
calibrate_threshold(labels, scores, part_ids=...) offers a labelled convenience
form. Calibration no longer maximizes F2 or normalizes each image by its own peak.
Use the same fixed scores and thresholds during live inspection and evaluation.
"""
from src.evaluation.calibration import (OperatingPoint, calibrate_operating_point,
                                       calibrate_threshold, localization_threshold, wilson_interval)

__all__ = ["OperatingPoint", "calibrate_operating_point", "calibrate_threshold",
           "localization_threshold", "wilson_interval"]
