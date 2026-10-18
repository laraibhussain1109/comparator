"""Lazy optional neural backends; geometry operation does not import PyTorch."""
from src.models.efficientad import EfficientADBackend
from src.models.registry import available_backends, create_backend
from src.models.weights import ModelSetupError

__all__ = ["EfficientADBackend", "ModelSetupError", "available_backends", "create_backend"]
