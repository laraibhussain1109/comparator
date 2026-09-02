from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class DefectRegion:
    kind: str
    contour: np.ndarray
    bounding_box: tuple[int, int, int, int]
    confidence: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "bounding_box": list(self.bounding_box), "confidence": round(self.confidence, 4), "source": self.source}


@dataclass
class InspectionResult:
    result: str
    registration_confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    geometry: dict[str, Any] = field(default_factory=dict)
    regions: list[DefectRegion] = field(default_factory=list)
    marked_image: np.ndarray | None = None
    part_present: bool = True
