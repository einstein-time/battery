"""Tests for neural network models."""

import torch
import pytest

from src.models import PCUNet, SimpleCNN


def test_pcunet_forward():
    """Test PC-UNet forward pass."""
    model = PCUNet(
        in_channels=9,
        out_channels=1,
        base_features=16,
        num_levels=3,
        dropout_rate=0.1,
    )

    # Create dummy input
    batch_size = 2
    H, W = 32, 32

    x = torch.randn(batch_size, 9, H, W)
    physics = torch.randn(batch_size, 3)

    # Forward pass
    output = model(x, physics)

    assert output.shape == (batch_size, 1, H, W)
    assert not torch.isnan(output).any()


def test_pcunet_no_physics():
    """Test PC-UNet forward pass without physics parameters."""
    model = PCUNet(in_channels=9, out_channels=1)

    x = torch.randn(2, 9, 32, 32)

    # Forward pass without physics
    output = model(x)

    assert output.shape == (2, 1, 32, 32)


def test_simple_cnn_forward():
    """Test SimpleCNN forward pass."""
    model = SimpleCNN(
        in_channels=9,
        out_channels=1,
        hidden_channels=32,
        num_layers=4,
    )

    x = torch.randn(2, 9, 32, 32)

    output = model(x)

    assert output.shape == (2, 1, 32, 32)
    assert not torch.isnan(output).any()


def test_model_parameter_count():
    """Test that parameter counting works."""
    model = PCUNet(in_channels=9, out_channels=1)

    n_params = model.count_parameters()

    assert n_params > 0
    assert isinstance(n_params, int)


def test_pcunet_uncertainty():
    """Test MC dropout uncertainty estimation."""
    model = PCUNet(
        in_channels=9,
        out_channels=1,
        dropout_rate=0.2,
    )

    x = torch.randn(1, 9, 32, 32)
    physics = torch.randn(1, 3)

    mean, std = model.predict_with_uncertainty(x, physics, n_samples=5)

    assert mean.shape == (1, 1, 32, 32)
    assert std.shape == (1, 1, 32, 32)
    assert (std >= 0).all()  # Std should be non-negative


def test_model_different_sizes():
    """Test models work with different input sizes."""
    model = PCUNet(in_channels=9, out_channels=1)

    for size in [16, 32, 64]:
        x = torch.randn(1, 9, size, size)
        output = model(x)
        assert output.shape == (1, 1, size, size)


@pytest.mark.parametrize("device_type", ["cpu"])
def test_model_on_device(device_type):
    """Test model works on different devices."""
    device = torch.device(device_type)

    model = PCUNet(in_channels=9, out_channels=1).to(device)
    x = torch.randn(1, 9, 32, 32).to(device)

    output = model(x)

    assert output.device.type == device_type
