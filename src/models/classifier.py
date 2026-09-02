from __future__ import annotations

CLASS_NAMES = {0: "GOOD", 1: "NG"}
ARCHITECTURE = "convnext_tiny_imagenet"


def validate_class_mapping(mapping: dict) -> None:
    normalized = {int(k): v for k, v in mapping.items()}
    if normalized != CLASS_NAMES:
        raise ValueError(f"Invalid class mapping {mapping}; required {CLASS_NAMES}")


def require_torch():
    try:
        import torch
        import torchvision
    except ImportError as exc:
        raise RuntimeError(
            "The supervised classifier requires PyTorch and torchvision. "
            "Install requirements.txt; full training has not started."
        ) from exc
    return torch, torchvision


def create_classifier(pretrained: bool = True, freeze_backbone: bool = True):
    torch, torchvision = require_torch()
    weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        model = torchvision.models.convnext_tiny(weights=weights)
    except Exception as exc:
        if pretrained:
            raise RuntimeError("Could not load pretrained ConvNeXt-Tiny weights; refusing to train a random backbone") from exc
        raise
    feature_dim = model.classifier[-1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(1), torch.nn.Dropout(.2), torch.nn.Linear(feature_dim, 256),
        torch.nn.GELU(), torch.nn.Dropout(.2), torch.nn.Linear(256, 2),
    )
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    return model


def assert_logits(logits) -> None:
    torch, _ = require_torch()
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise RuntimeError(f"Classifier must return [batch, 2] raw logits, got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Classifier produced non-finite logits")
