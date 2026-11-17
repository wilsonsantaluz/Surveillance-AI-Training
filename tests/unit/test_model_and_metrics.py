import pytest
import torch

from surveillance_ml.models.classifier import build_resnet18
from surveillance_ml.training.metrics import classification_metrics


def test_resnet18_head_strategy_replaces_classifier_and_freezes_backbone() -> None:
    model = build_resnet18(num_classes=3, strategy="head")

    assert model.fc.out_features == 3
    assert all(not parameter.requires_grad for parameter in model.layer1.parameters())
    assert all(parameter.requires_grad for parameter in model.fc.parameters())


def test_resnet18_partial_strategy_unfreezes_last_stage() -> None:
    model = build_resnet18(num_classes=3, strategy="partial")

    assert all(parameter.requires_grad for parameter in model.layer4.parameters())
    assert all(not parameter.requires_grad for parameter in model.layer3.parameters())


def test_metrics_include_zero_support_classes() -> None:
    result = classification_metrics([0, 1], [0, 0], average_loss=1.2, num_classes=3)

    assert result.accuracy == 0.5
    assert result.confusion_matrix == [[1, 0, 0], [1, 0, 0], [0, 0, 0]]
    assert result.precision == pytest.approx(1 / 6)
    assert result.recall == pytest.approx(1 / 3)
    assert result.f1 == pytest.approx(2 / 9)


def test_model_forward_shape() -> None:
    model = build_resnet18(num_classes=3, strategy="head")
    output = model(torch.zeros((2, 3, 32, 32)))

    assert output.shape == (2, 3)
