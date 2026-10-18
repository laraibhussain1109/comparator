"""Physical-part inspection sessions replace frame-vote verdict filtering.

Migration: record ViewDecision instances in an InspectionSession. Repeated frames
are not independent defect evidence; final PASS requires every mandatory view.
Capture stability belongs to src.acquisition.quality rather than verdict voting.
"""
from src.inspection.session import InspectionSession, SessionState, ViewDecision

__all__ = ["InspectionSession", "SessionState", "ViewDecision"]
