"""Configurable TorchVision ResNet18 scene classifier."""

from typing import cast

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.resnet import ResNet


def build_resnet18(
    num_classes: int,
    strategy: str = "full",
    pretrained: bool = False,
) -> nn.Module:
    """Build ResNet18 with head-only, partial, or full fine-tuning."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if strategy not in {"head", "partial", "full"}:
        raise ValueError("strategy must be one of: head, partial, full")
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = cast(ResNet, resnet18(weights=weights))
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    for parameter in model.parameters():
        parameter.requires_grad = strategy == "full"
    if strategy == "partial":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return cast(nn.Module, model)
