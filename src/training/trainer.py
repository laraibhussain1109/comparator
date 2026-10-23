"""Public trainer entry point for the physical-part-separated NeuroIris workflow."""
from __future__ import annotations
from typing import Callable


class Trainer:
    def __init__(self, config: dict, progress: Callable | None = None):
        self.config = config
        self.progress = progress

    def run(self, view: str = 'inner_face', backend: str | None = None) -> dict:
        from src.service import NeuroIrisService
        return NeuroIrisService(self.config).train(view, backend or self.config['models']['backend'], self.progress)
