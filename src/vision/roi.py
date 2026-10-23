"""Serializable native-image ROI masks.

Coordinates are normalized to [0, 1] unless ``normalized: false`` is supplied.
Definitions accept rectangle (``rect: [x,y,width,height]``), circle
(``center: [x,y], radius: r``; radius relative to the shorter image dimension),
or polygon (``points: [[x,y], ...]``). ``mode: exclude`` removes pixels.
With no include definitions the initial mask is the complete image. Intersect
the returned mask with a HousingLocator mask to restrict processing to the part.
"""
from __future__ import annotations

import cv2
import numpy as np


def roi_mask(shape: tuple, definitions: list[dict] | None = None) -> np.ndarray:
    h, w = shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("ROI dimensions must be positive")
    definitions = definitions or []
    def is_exclusion(d: dict) -> bool:
        return bool(d.get("exclusion", False)) or d.get("mode") == "exclude"
    includes = [d for d in definitions if not is_exclusion(d)]
    result = np.zeros((h, w), np.uint8) if includes else np.full((h, w), 255, np.uint8)
    # Exclusions always win, irrespective of the order in the saved definitions.
    for definition in includes + [d for d in definitions if is_exclusion(d)]:
        layer = np.zeros((h, w), np.uint8)
        normalized = definition.get("normalized", True)
        sx, sy = (w - 1, h - 1) if normalized else (1, 1)
        kind = definition.get("shape", definition.get("type", "rectangle")).lower()
        if kind in ("rectangle", "rect"):
            values = definition.get("rect", definition.get("bounds"))
            if values is None and "points" in definition:
                (x1, y1), (x2, y2) = definition["points"]
                values = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]
            if values is None:
                values = [definition[k] for k in ("x", "y", "width", "height")]
            x, y, rw, rh = map(float, values)
            cv2.rectangle(layer, (round(x * sx), round(y * sy)),
                          (round((x + rw) * sx), round((y + rh) * sy)), 255, -1)
        elif kind == "circle":
            if "points" in definition:
                (x, y), (ex, ey) = definition["points"]
                radius = float(np.hypot((ex - x) * sx, (ey - y) * sy))
            else:
                x, y = map(float, definition["center"])
                radius = float(definition["radius"]) * (min(h, w) - 1 if normalized else 1)
            if radius <= 0:
                raise ValueError("Circle ROI radius must be positive")
            cv2.circle(layer, (round(x * sx), round(y * sy)), round(radius), 255, -1)
        elif kind == "polygon":
            points = np.asarray(definition["points"], np.float64)
            if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
                raise ValueError("Polygon ROI requires at least three x/y points")
            cv2.fillPoly(layer, [np.rint(points * [sx, sy]).astype(np.int32)], 255)
        else:
            raise ValueError(f"Unsupported ROI shape: {kind}")
        if is_exclusion(definition):
            result[layer != 0] = 0
        else:
            result = cv2.bitwise_or(result, layer)
    return result
