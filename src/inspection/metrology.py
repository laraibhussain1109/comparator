"""Optional calibrated measurement boundary; no measurement hardware is simulated.

The current vision pipeline reports pixels. Scale conversion is an explicit
operation requiring a validated calibration for the camera and native image
size. A scalar scale does not establish micrometre accuracy, flatness or depth.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


@dataclass(frozen=True)
class Measurement:
    name: str
    value: float
    unit: str = "px"
    uncertainty: float | None = None
    calibration_id: str | None = None
    source: str = "vision"


@dataclass(frozen=True)
class ScaleCalibration:
    calibration_id: str
    camera_id: str
    image_size: tuple[int, int]  # width, height at calibration
    pixels_per_mm: float
    validated: bool = False
    scale_relative_uncertainty: float | None = None

    def to_mm(self, measurement: Measurement, *, camera_id: str,
              image_size: tuple[int, int]) -> Measurement:
        if not self.validated or not self.calibration_id or not self.camera_id:
            raise ValueError("A real validated calibration is required for mm measurements")
        if camera_id != self.camera_id or tuple(image_size) != tuple(self.image_size):
            raise ValueError("Calibration does not match the camera and native image size")
        if not math.isfinite(self.pixels_per_mm) or self.pixels_per_mm <= 0:
            raise ValueError("Calibration scale must be finite and positive")
        if measurement.unit != "px" or not math.isfinite(measurement.value):
            raise ValueError("Scale conversion requires a finite pixel measurement")
        uncertainty = None
        relative = self.scale_relative_uncertainty
        if relative is not None and (not math.isfinite(relative) or relative < 0):
            raise ValueError("Calibration uncertainty must be finite and nonnegative")
        if measurement.uncertainty is not None:
            if not math.isfinite(measurement.uncertainty) or measurement.uncertainty < 0:
                raise ValueError("Measurement uncertainty must be finite and nonnegative")
            # Report uncertainty only when both measurement and scale uncertainty
            # were supplied; a missing scale uncertainty is not an exact scale.
            if relative is not None:
                uncertainty = math.hypot(measurement.uncertainty / self.pixels_per_mm,
                                         measurement.value / self.pixels_per_mm * relative)
        return Measurement(measurement.name, measurement.value / self.pixels_per_mm,
                           "mm", uncertainty, self.calibration_id, measurement.source)


class MetrologyProvider(Protocol):
    @property
    def available(self) -> bool: ...

    def measure(self, part_id: str) -> list[Measurement]: ...


class DisabledMetrologyProvider:
    available = False
    reason = "No real metrology hardware and calibration are configured"

    def measure(self, part_id: str) -> list[Measurement]:
        raise RuntimeError(self.reason)


class LightingController(Protocol):
    @property
    def programmable(self) -> bool: ...

    def select_condition(self, condition: str) -> None: ...


class ConstantExternalLighting:
    programmable = False
    condition = "constant_external"

    def select_condition(self, condition: str) -> None:
        if condition != self.condition:
            raise RuntimeError("Programmable lighting hardware is not configured")
