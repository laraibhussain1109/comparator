"""Honest capability registry for optional, independently selectable models."""
from __future__ import annotations

from importlib import metadata, util
from typing import Any

from src.models.weights import ModelSetupError


def available_backends() -> dict[str, dict[str, Any]]:
    installed = util.find_spec("torch") is not None and util.find_spec("anomalib") is not None
    try:
        version = metadata.version("anomalib") if installed else None
    except metadata.PackageNotFoundError:
        version = None
    compatible = installed and version == "2.2.0"
    return {
        "geometry": {"available": True, "reason": "OpenCV geometry backend; trained view profile required."},
        "efficientad": {"available": compatible, "package_version": version,
                        "reason": "Pinned software installed; teacher assets, trained view and hardware checked when used."
                        if compatible else "Install the pinned optional neural environment (requirements-neural.txt)."},
        "dinomaly": {"available": False, "reason": "No Dinomaly adapter has been implemented and validated in this release."},
        "dinov2_reference": {"available": False, "reason": "DINOv2 reference comparison has not been implemented or benchmarked."},
    }


def create_backend(name: str, settings: dict | None = None):
    key = name.strip().lower()
    if key == "efficientad":
        from src.models.efficientad import EfficientADBackend
        return EfficientADBackend(settings)
    if key in {"dinomaly", "dinov2", "dinov2_reference"}:
        raise ModelSetupError(f"{name} is optional and unavailable: no validated adapter is installed. Select EfficientAD or geometry.")
    raise ValueError(f"Unknown neural backend: {name}")
