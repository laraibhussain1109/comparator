"""Localized, calibrated evidence fusion.

Migration: EvidenceDecisionEngine consumes measured geometry and anomaly
candidates with independent calibration. The old SSIM-supported, image-normalized
DefectLocalizationFusion is retired. Native-to-source candidate mapping and
rendering are provided by src.inspection.view_pipeline.
"""
from src.inspection.fusion import EvidenceDecisionEngine
from src.inspection.session import ViewDecision

__all__ = ["EvidenceDecisionEngine", "ViewDecision"]
