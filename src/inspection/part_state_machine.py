"""Explicit multi-view physical-part state and decisions.

Migration: InspectionSession tracks one part ID and required views. It cannot
produce final PASS while any required view is missing or unresolved. The old
single-frame presence/GOOD/NG PartStateMachine API is retired.
"""
from src.inspection.session import InspectionSession, SessionState, ViewDecision

__all__ = ["InspectionSession", "SessionState", "ViewDecision"]
