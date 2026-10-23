"""Calibrated and localized evidence decisions.

Migration: use EvidenceDecisionEngine with valid registration, independent
GOOD-part calibration, and localized evidence. The old score-only DecisionEngine
is retired. Aggregate mandatory views with InspectionSession before final PASS.
"""
from src.inspection.fusion import EvidenceDecisionEngine
from src.inspection.session import InspectionSession, SessionState, ViewDecision

__all__ = ["EvidenceDecisionEngine", "InspectionSession", "SessionState", "ViewDecision"]
