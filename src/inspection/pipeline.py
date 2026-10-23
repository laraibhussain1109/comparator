"""Public single-view pipeline using the same decision path as live inspection.

For final physical-part decisions use NeuroIrisService/InspectionSession, which
requires every configured mandatory view before PASS.
"""
from __future__ import annotations


class InspectionPipeline:
    def __init__(self, config: dict, view: str = 'inner_face', backend: str | None = None):
        from src.service import NeuroIrisService
        self.service = NeuroIrisService(config)
        self.view = view
        self.backend = backend or config['models']['backend']

    def inspect(self, frame, view: str | None = None):
        decision, marked, _ = self.service._infer(frame, view or self.view, self.backend)
        return {**decision.to_dict(), 'marked_image': marked}
