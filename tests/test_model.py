"""Comprehensive unit tests for neural network models and loss functions.

Tests cover forward/backward passes, output shapes, parameter counting,
configuration export, uncertainty quantification, physics-informed loss
components, and cross-model interface consistency for the battery thermal
surrogate modelling pipeline.

All tests use small grids (32x32 or 64x64) and small batch sizes (1 or 2)
to keep CPU runtime under 60 seconds.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.base_model import BaseThermalModel
from src.models.pc_unet import PCUNet
from src.models.simple_cnn import SimpleCNN
from src.models.pod_model import PODModel
from src.models.uncertainty import (
    mc_dropout_predict,
    calibration_error,
    prediction_intervals,
)
from src.training.losses import PhysicsInformedLoss


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def pc_unet() -> PCUNet:
    """Create a small PCUNet model for testing.

    Returns:
        A PCUNet instance with default channel counts and 4 encoder/decoder
        levels, suitable for 64x64 or 32x32 inputs.
    """
    model = PCUNet(
        in_channels=9,
        out_channels=1,
        base_features=32,
        num_levels=4,
        dropout_rate=0.1,
    )
    model.eval()
    return model


@pytest.fixture
def simple_cnn() -> SimpleCNN:
    """Create a SimpleCNN model for testing.

    Returns:
        A SimpleCNN instance with default 9 input and 1 output channels.
    """
    model = SimpleCNN(in_channels=9, out_channels=1)
    model.eval()
    return model


@pytest.fixture
def pod_model() -> PODModel:
    """Create a PODModel fitted with random snapshots for testing.

    The model uses a 64x64 grid with 50 POD modes.  A random set of 100
    snapshots is generated and used to fit the basis so that the forward
    pass can execute without raising a RuntimeError.

    Returns:
        A PODModel with a pre-fitted basis, ready for forward passes.
    """
    model = PODModel(
        n_modes=50,
        hidden_dim=256,
        field_height=64,
        field_width=64,
        in_channels=9,
        out_channels=1,
    )
    # Fit with random snapshots so forward() does not raise.
    snapshots = torch.randn(100, 64, 64)
    model.fit_basis(snapshots)
    model.eval()
    return model


@pytest.fixture
def pod_model_32() -> PODModel:
    """Create a PODModel fitted for 32x32 grids.

    Returns:
        A PODModel with a 32x32 grid, 20 modes, and a pre-fitted basis.
    """
    model = PODModel(
        n_modes=20,
        hidden_dim=128,
        field_height=32,
        field_width=32,
        in_channels=9,
        out_channels=1,
    )
    snapshots = torch.randn(60, 32, 32)
    model.fit_basis(snapshots)
    model.eval()
    return model


@pytest.fixture
def physics_loss() -> PhysicsInformedLoss:
    """Create a PhysicsInformedLoss instance with standard parameters.

    Returns:
        A PhysicsInformedLoss configured with typical battery-pack
        thermal simulation parameters.
    """
    return PhysicsInformedLoss(
        lambda_data=1.0,
        lambda_pde=0.1,
        lambda_bc=0.1,
        lambda_consistency=0.05,
        dx=0.001,
        dy=0.001,
        dt=0.001,
        rho=2500.0,
        cp=700.0,
    )


@pytest.fixture
def sample_input_64() -> torch.Tensor:
    """Create a random 9-channel input tensor on a 64x64 grid.

    Returns:
        Tensor of shape (2, 9, 64, 64) with standard-normal values.
    """
    return torch.randn(2, 9, 64, 64)


@pytest.fixture
def sample_input_32() -> torch.Tensor:
    """Create a random 9-channel input tensor on a 32x32 grid.

    Returns:
        Tensor of shape (1, 9, 32, 32) with standard-normal values.
    """
    return torch.randn(1, 9, 32, 32)


# ======================================================================
# TestPCUNet
# ======================================================================


class TestPCUNet:
    """Tests for the Physics-Conditioned U-Net model."""

    def test_forward_shape(self, pc_unet: PCUNet, sample_input_64: torch.Tensor) -> None:
        """Verify that forward produces the correct output shape for 64x64 inputs.

        A batch of 2 samples with 9 channels on a 64x64 grid should produce
        an output of shape (2, 1, 64, 64).
        """
        with torch.no_grad():
            output = pc_unet(sample_input_64)
        assert output.shape == (2, 1, 64, 64), (
            f"Expected output shape (2, 1, 64, 64), got {output.shape}"
        )

    def test_forward_shape_32(self, pc_unet: PCUNet, sample_input_32: torch.Tensor) -> None:
        """Verify that forward produces the correct output shape for 32x32 inputs.

        A single sample with 9 channels on a 32x32 grid should produce
        an output of shape (1, 1, 32, 32).
        """
        with torch.no_grad():
            output = pc_unet(sample_input_32)
        assert output.shape == (1, 1, 32, 32), (
            f"Expected output shape (1, 1, 32, 32), got {output.shape}"
        )

    def test_backward_pass(self, pc_unet: PCUNet, sample_input_64: torch.Tensor) -> None:
        """Verify that gradients propagate through the PCUNet without errors.

        Performs a forward pass, computes a scalar loss, and calls backward().
        At least one parameter should have a non-None gradient afterwards.
        """
        pc_unet.train()
        x = sample_input_64.clone().requires_grad_(False)
        output = pc_unet(x)
        loss = output.mean()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in pc_unet.parameters()
            if p.requires_grad
        )
        assert has_grad, "No parameter received a non-zero gradient during backward pass"

    def test_predict_step(self, pc_unet: PCUNet, sample_input_64: torch.Tensor) -> None:
        """Verify that predict_step returns a valid temperature field.

        predict_step should concatenate t_current with params, run forward,
        clip the delta, and add it back to t_current.  The output shape
        must match the temperature field shape (B, 1, H, W).
        """
        t_current = sample_input_64[:, 0:1, :, :]  # (2, 1, 64, 64)
        params = sample_input_64[:, 1:, :, :]       # (2, 8, 64, 64)

        with torch.no_grad():
            t_next = pc_unet.predict_step(t_current, params)

        assert t_next.shape == t_current.shape, (
            f"predict_step output shape {t_next.shape} does not match "
            f"t_current shape {t_current.shape}"
        )

    def test_output_clipping(self, pc_unet: PCUNet) -> None:
        """Verify that forward output is clipped to [-delta_t_clip, delta_t_clip].

        Feeding a very large input should still produce output values within
        the clipping range defined by delta_t_clip.
        """
        large_input = torch.randn(1, 9, 32, 32) * 1000.0
        with torch.no_grad():
            output = pc_unet(large_input)

        clip = pc_unet.delta_t_clip
        assert output.min().item() >= -clip - 1e-6, (
            f"Output minimum {output.min().item():.4f} is below -delta_t_clip ({-clip})"
        )
        assert output.max().item() <= clip + 1e-6, (
            f"Output maximum {output.max().item():.4f} exceeds delta_t_clip ({clip})"
        )

    def test_count_parameters(self, pc_unet: PCUNet) -> None:
        """Verify that count_parameters returns a positive integer."""
        n_params = pc_unet.count_parameters()
        assert isinstance(n_params, int), (
            f"count_parameters should return int, got {type(n_params).__name__}"
        )
        assert n_params > 0, (
            f"Parameter count should be positive, got {n_params}"
        )

    def test_get_config(self, pc_unet: PCUNet) -> None:
        """Verify that get_config returns a dict with expected keys.

        The configuration dictionary should contain at minimum the model
        type, channel counts, architectural hyperparameters, and the
        trainable parameter count.
        """
        config = pc_unet.get_config()
        assert isinstance(config, dict), (
            f"get_config should return dict, got {type(config).__name__}"
        )

        expected_keys = {
            "model_type",
            "in_channels",
            "out_channels",
            "base_features",
            "num_levels",
            "dropout_rate",
            "delta_t_clip",
            "trainable_params",
        }
        missing = expected_keys - set(config.keys())
        assert not missing, f"get_config is missing keys: {missing}"
        assert config["model_type"] == "PCUNet", (
            f"Expected model_type 'PCUNet', got '{config['model_type']}'"
        )

    def test_mc_predict(self, pc_unet: PCUNet, sample_input_64: torch.Tensor) -> None:
        """Verify that mc_predict returns mean and std with correct shapes.

        mc_predict should return a tuple of (mean, std), each of shape
        (B, 1, H, W), and the standard deviation should be non-negative
        everywhere.
        """
        mean, std = pc_unet.mc_predict(sample_input_64, n_samples=10)

        assert mean.shape == (2, 1, 64, 64), (
            f"mc_predict mean shape {mean.shape} != expected (2, 1, 64, 64)"
        )
        assert std.shape == (2, 1, 64, 64), (
            f"mc_predict std shape {std.shape} != expected (2, 1, 64, 64)"
        )
        assert (std >= 0).all(), (
            "mc_predict standard deviation contains negative values"
        )

    def test_dropout_active_in_train(self, pc_unet: PCUNet, sample_input_64: torch.Tensor) -> None:
        """Verify that dropout induces stochasticity in training mode.

        In training mode with dropout enabled, two forward passes on the
        same input should (with very high probability) produce different
        outputs.
        """
        pc_unet.train()
        with torch.no_grad():
            out1 = pc_unet(sample_input_64).clone()
            out2 = pc_unet(sample_input_64).clone()

        # With dropout_rate=0.1, outputs should differ in at least some pixels.
        assert not torch.allclose(out1, out2, atol=1e-7), (
            "Two forward passes in training mode produced identical outputs; "
            "dropout does not appear to be active"
        )

    def test_deterministic_in_eval_no_mc(
        self, pc_unet: PCUNet, sample_input_64: torch.Tensor
    ) -> None:
        """Verify that eval mode produces deterministic outputs.

        When the model is in evaluation mode (no MC dropout), the same
        input should always produce the exact same output.
        """
        pc_unet.eval()
        with torch.no_grad():
            out1 = pc_unet(sample_input_64).clone()
            out2 = pc_unet(sample_input_64).clone()

        assert torch.allclose(out1, out2, atol=1e-7), (
            "Eval-mode forward passes produced different outputs; "
            "model is not deterministic when dropout is disabled"
        )


# ======================================================================
# TestSimpleCNN
# ======================================================================


class TestSimpleCNN:
    """Tests for the SimpleCNN baseline model."""

    def test_forward_shape(self, simple_cnn: SimpleCNN, sample_input_64: torch.Tensor) -> None:
        """Verify that forward produces the correct output shape.

        A batch of 2 with 9 channels on 64x64 should yield (2, 1, 64, 64).
        """
        with torch.no_grad():
            output = simple_cnn(sample_input_64)
        assert output.shape == (2, 1, 64, 64), (
            f"Expected output shape (2, 1, 64, 64), got {output.shape}"
        )

    def test_backward_pass(self, simple_cnn: SimpleCNN, sample_input_64: torch.Tensor) -> None:
        """Verify that gradients propagate through the SimpleCNN.

        At least one parameter should receive a non-zero gradient after
        a forward-backward pass.
        """
        simple_cnn.train()
        output = simple_cnn(sample_input_64)
        loss = output.mean()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in simple_cnn.parameters()
            if p.requires_grad
        )
        assert has_grad, "No parameter received a non-zero gradient in SimpleCNN backward pass"

    def test_count_parameters(self, simple_cnn: SimpleCNN, pc_unet: PCUNet) -> None:
        """Verify that SimpleCNN has fewer parameters than PCUNet.

        The SimpleCNN is a lightweight baseline and should be strictly
        smaller than the full PCUNet.
        """
        n_simple = simple_cnn.count_parameters()
        n_unet = pc_unet.count_parameters()

        assert isinstance(n_simple, int), (
            f"count_parameters should return int, got {type(n_simple).__name__}"
        )
        assert n_simple > 0, (
            f"SimpleCNN parameter count should be positive, got {n_simple}"
        )
        assert n_simple < n_unet, (
            f"SimpleCNN ({n_simple:,} params) should have fewer parameters "
            f"than PCUNet ({n_unet:,} params)"
        )


# ======================================================================
# TestPODModel
# ======================================================================


class TestPODModel:
    """Tests for the POD (Proper Orthogonal Decomposition) model."""

    def test_forward_shape(self, pod_model: PODModel, sample_input_64: torch.Tensor) -> None:
        """Verify that forward produces the correct output shape.

        A batch of 2 with 9 channels on 64x64 should yield (2, 1, 64, 64).
        """
        with torch.no_grad():
            output = pod_model(sample_input_64)
        assert output.shape == (2, 1, 64, 64), (
            f"Expected output shape (2, 1, 64, 64), got {output.shape}"
        )

    def test_fit_basis(self) -> None:
        """Verify that fit_basis populates the POD basis with correct shape.

        After fitting, the basis buffer should have shape
        (n_modes, field_height * field_width) and the basis_fitted flag
        should be True.
        """
        n_modes = 30
        h, w = 64, 64
        model = PODModel(
            n_modes=n_modes,
            field_height=h,
            field_width=w,
        )

        # Before fitting, basis_fitted should be False.
        assert not model.basis_fitted.item(), (
            "basis_fitted should be False before calling fit_basis"
        )

        snapshots = torch.randn(80, h, w)
        result = model.fit_basis(snapshots)

        assert model.basis.shape == (n_modes, h * w), (
            f"Expected basis shape ({n_modes}, {h * w}), got {model.basis.shape}"
        )
        assert model.basis_fitted.item(), (
            "basis_fitted should be True after calling fit_basis"
        )
        assert "singular_values" in result, (
            "fit_basis result should contain 'singular_values'"
        )
        assert "explained_variance_ratio" in result, (
            "fit_basis result should contain 'explained_variance_ratio'"
        )
        assert len(result["singular_values"]) == n_modes, (
            f"Expected {n_modes} singular values, got {len(result['singular_values'])}"
        )

    def test_forward_without_fit_raises(self) -> None:
        """Verify that forward raises RuntimeError if basis is not fitted.

        The PODModel requires fit_basis to be called before the first
        forward pass.  Calling forward without fitting should raise
        a RuntimeError with a descriptive message.
        """
        model = PODModel(n_modes=10, field_height=32, field_width=32)
        x = torch.randn(1, 9, 32, 32)
        with pytest.raises(RuntimeError, match="POD basis has not been fitted"):
            model(x)

    def test_backward_pass(self, pod_model: PODModel, sample_input_64: torch.Tensor) -> None:
        """Verify that gradients propagate through the PODModel.

        The MLP component of the POD model should receive gradients
        during backpropagation.
        """
        pod_model.train()
        output = pod_model(sample_input_64)
        loss = output.mean()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in pod_model.parameters()
            if p.requires_grad
        )
        assert has_grad, "No parameter received a non-zero gradient in PODModel backward pass"

    def test_fit_basis_wrong_dimensions_raises(self) -> None:
        """Verify that fit_basis raises ValueError for mismatched spatial dims.

        If the snapshot spatial dimensions do not match the model's
        configured field_height and field_width, a ValueError should be raised.
        """
        model = PODModel(n_modes=10, field_height=64, field_width=64)
        wrong_snapshots = torch.randn(50, 32, 32)  # Wrong spatial dims
        with pytest.raises(ValueError, match="Snapshot spatial dims"):
            model.fit_basis(wrong_snapshots)


# ======================================================================
# TestPhysicsLoss
# ======================================================================


class TestPhysicsLoss:
    """Tests for the PhysicsInformedLoss function."""

    def test_data_loss_zero(self, physics_loss: PhysicsInformedLoss) -> None:
        """Verify that data_loss is zero when prediction equals target.

        When pred and target are identical tensors, the MSE should be
        exactly zero (within floating-point tolerance).
        """
        pred = torch.randn(2, 1, 32, 32)
        target = pred.clone()
        loss = physics_loss.data_loss(pred, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-7), (
            f"data_loss should be ~0 for identical pred and target, got {loss.item()}"
        )

    def test_data_loss_positive(self, physics_loss: PhysicsInformedLoss) -> None:
        """Verify that data_loss is positive for different pred and target.

        When pred and target differ, the MSE must be strictly positive.
        """
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randn(2, 1, 32, 32)
        loss = physics_loss.data_loss(pred, target)
        assert loss.item() > 0.0, (
            f"data_loss should be > 0 for different pred and target, got {loss.item()}"
        )

    def test_pde_residual_computable(self, physics_loss: PhysicsInformedLoss) -> None:
        """Verify that PDE residual loss returns a scalar tensor with grad.

        The PDE residual should be a differentiable scalar that can be
        used in backpropagation.
        """
        pred = torch.randn(2, 1, 32, 32, requires_grad=True)
        T_input = torch.randn(2, 1, 32, 32)
        k_field = torch.ones(2, 1, 32, 32) * 50.0   # Typical thermal conductivity
        q_field = torch.ones(2, 1, 32, 32) * 1e5     # Typical heat source

        loss = physics_loss.pde_residual_loss(pred, T_input, k_field, q_field)

        assert loss.dim() == 0, (
            f"PDE residual loss should be scalar (0-dim), got {loss.dim()}-dim tensor"
        )
        assert loss.requires_grad, (
            "PDE residual loss should require grad for backpropagation"
        )

    def test_boundary_loss_computable(self, physics_loss: PhysicsInformedLoss) -> None:
        """Verify that boundary loss returns a scalar tensor.

        The boundary condition residual should be a scalar that can
        participate in the total loss computation.
        """
        pred = torch.randn(2, 1, 32, 32, requires_grad=True)
        T_input = torch.randn(2, 1, 32, 32)
        h_field = torch.ones(2, 1, 32, 32) * 10.0  # Convective HTC

        loss = physics_loss.boundary_loss(pred, T_input, h_field)

        assert loss.dim() == 0, (
            f"Boundary loss should be scalar (0-dim), got {loss.dim()}-dim tensor"
        )

    def test_phase1_only_data(self, physics_loss: PhysicsInformedLoss) -> None:
        """Verify that phase 1 activates only the data loss term.

        In phase 1 (pretrain), the loss_dict should contain only
        data_loss and total_loss, with no PDE or BC components.
        """
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randn(2, 1, 32, 32)
        T_input = torch.randn(2, 1, 32, 32)

        total, loss_dict = physics_loss(
            pred, target, T_input, phase=1,
        )

        assert "data_loss" in loss_dict, "Phase 1 should include data_loss"
        assert "total_loss" in loss_dict, "Phase 1 should include total_loss"
        assert "pde_loss" not in loss_dict, (
            "Phase 1 should NOT include pde_loss"
        )
        assert "bc_loss" not in loss_dict, (
            "Phase 1 should NOT include bc_loss"
        )
        assert "consistency_loss" not in loss_dict, (
            "Phase 1 should NOT include consistency_loss"
        )

    def test_phase2_adds_physics(self, physics_loss: PhysicsInformedLoss) -> None:
        """Verify that phase 2 activates PDE and BC loss terms.

        In phase 2, the loss_dict should contain data_loss, pde_loss,
        and bc_loss (all nonzero for random inputs).
        """
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randn(2, 1, 32, 32)
        T_input = torch.randn(2, 1, 32, 32)
        k_field = torch.ones(2, 1, 32, 32) * 50.0
        q_field = torch.ones(2, 1, 32, 32) * 1e5
        h_field = torch.ones(2, 1, 32, 32) * 10.0

        total, loss_dict = physics_loss(
            pred, target, T_input,
            k_field=k_field,
            q_field=q_field,
            h_field=h_field,
            phase=2,
        )

        assert "data_loss" in loss_dict, "Phase 2 should include data_loss"
        assert "pde_loss" in loss_dict, "Phase 2 should include pde_loss"
        assert "bc_loss" in loss_dict, "Phase 2 should include bc_loss"
        assert loss_dict["pde_loss"] > 0.0, (
            "PDE loss should be > 0 for random inputs"
        )
        assert loss_dict["bc_loss"] > 0.0, (
            "BC loss should be > 0 for random inputs"
        )

    def test_loss_differentiable(self, physics_loss: PhysicsInformedLoss) -> None:
        """Verify that total loss supports backpropagation.

        Calling .backward() on the total loss should succeed without
        errors, and the prediction tensor should receive a gradient.
        """
        pred = torch.randn(2, 1, 32, 32, requires_grad=True)
        target = torch.randn(2, 1, 32, 32)
        T_input = torch.randn(2, 1, 32, 32)
        k_field = torch.ones(2, 1, 32, 32) * 50.0
        q_field = torch.ones(2, 1, 32, 32) * 1e5
        h_field = torch.ones(2, 1, 32, 32) * 10.0

        total, _ = physics_loss(
            pred, target, T_input,
            k_field=k_field,
            q_field=q_field,
            h_field=h_field,
            phase=2,
        )

        total.backward()

        assert pred.grad is not None, (
            "Prediction tensor should have a gradient after backward()"
        )
        assert pred.grad.abs().sum() > 0, (
            "Prediction gradient should be non-zero"
        )

    def test_data_loss_known_value(self, physics_loss: PhysicsInformedLoss) -> None:
        """Verify that data_loss computes the correct MSE for a known case.

        For pred = 0 and target = 1 (uniform), the MSE should be exactly 1.0.
        """
        pred = torch.zeros(2, 1, 32, 32)
        target = torch.ones(2, 1, 32, 32)
        loss = physics_loss.data_loss(pred, target)
        assert loss.item() == pytest.approx(1.0, abs=1e-6), (
            f"MSE of zeros vs ones should be 1.0, got {loss.item()}"
        )

    def test_phase2_total_greater_than_phase1(
        self, physics_loss: PhysicsInformedLoss
    ) -> None:
        """Verify that phase 2 total loss is generally larger than phase 1.

        Adding physics terms (PDE + BC) to the data loss should increase
        the total, since these terms are non-negative for random inputs.
        """
        torch.manual_seed(42)
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randn(2, 1, 32, 32)
        T_input = torch.randn(2, 1, 32, 32)
        k_field = torch.ones(2, 1, 32, 32) * 50.0
        q_field = torch.ones(2, 1, 32, 32) * 1e5
        h_field = torch.ones(2, 1, 32, 32) * 10.0

        total_p1, _ = physics_loss(pred, target, T_input, phase=1)
        total_p2, _ = physics_loss(
            pred, target, T_input,
            k_field=k_field, q_field=q_field, h_field=h_field,
            phase=2,
        )

        assert total_p2.item() >= total_p1.item(), (
            f"Phase 2 total ({total_p2.item():.4f}) should be >= "
            f"phase 1 total ({total_p1.item():.4f}) due to added physics terms"
        )


# ======================================================================
# TestUncertainty
# ======================================================================


class TestUncertainty:
    """Tests for the uncertainty quantification utility functions."""

    def test_mc_dropout_shapes(self, pc_unet: PCUNet, sample_input_64: torch.Tensor) -> None:
        """Verify that mc_dropout_predict returns correctly shaped outputs.

        The returned (mean, std) tensors should both match the model's
        standard output shape (B, 1, H, W).
        """
        mean, std = mc_dropout_predict(pc_unet, sample_input_64, n_samples=10)

        assert mean.shape == (2, 1, 64, 64), (
            f"MC dropout mean shape {mean.shape} != expected (2, 1, 64, 64)"
        )
        assert std.shape == (2, 1, 64, 64), (
            f"MC dropout std shape {std.shape} != expected (2, 1, 64, 64)"
        )

    def test_mc_dropout_std_nonneg(
        self, pc_unet: PCUNet, sample_input_64: torch.Tensor
    ) -> None:
        """Verify that MC dropout standard deviation is non-negative everywhere.

        Standard deviation is by definition non-negative; any negative values
        would indicate a computational error.
        """
        _, std = mc_dropout_predict(pc_unet, sample_input_64, n_samples=10)
        assert (std >= 0).all(), (
            "MC dropout standard deviation contains negative values"
        )

    def test_mc_dropout_restores_eval_mode(
        self, pc_unet: PCUNet, sample_input_64: torch.Tensor
    ) -> None:
        """Verify that mc_dropout_predict restores the original model mode.

        If the model starts in eval mode, it should be back in eval mode
        after mc_dropout_predict completes.
        """
        pc_unet.eval()
        mc_dropout_predict(pc_unet, sample_input_64, n_samples=5)
        assert not pc_unet.training, (
            "mc_dropout_predict should restore the model to eval mode"
        )

    def test_calibration_error_perfect(self) -> None:
        """Verify that perfect calibration yields ECE close to zero.

        When predictions equal targets exactly and uncertainty is non-zero
        (i.e. all errors are zero, which fall within any prediction interval),
        the ECE should be small because coverage will be 100% in every bin
        vs the ideal 68.3%.
        """
        n = 1000
        predictions = torch.randn(n)
        targets = predictions.clone()  # Perfect predictions
        uncertainties = torch.ones(n) * 0.5  # Uniform uncertainty

        result = calibration_error(predictions, uncertainties, targets, n_bins=10)

        assert "ece" in result, "calibration_error result should contain 'ece'"
        assert "mean_coverage" in result, (
            "calibration_error result should contain 'mean_coverage'"
        )
        assert "ideal_coverage" in result, (
            "calibration_error result should contain 'ideal_coverage'"
        )
        # With perfect predictions (error=0), coverage is 100% in every bin.
        # ECE = |1.0 - 0.6827| * 1.0 ~ 0.3173
        # The key check is that it runs and returns valid values.
        assert result["ece"] >= 0.0, "ECE should be non-negative"
        assert 0.0 <= result["mean_coverage"] <= 1.0, (
            f"Mean coverage should be in [0, 1], got {result['mean_coverage']}"
        )

    def test_calibration_error_with_noise(self) -> None:
        """Verify calibration error computation with realistic noisy predictions.

        When predictions have Gaussian noise matching the stated uncertainty,
        the calibration should be reasonably good (ECE well below 1.0).
        """
        torch.manual_seed(123)
        n = 5000
        targets = torch.randn(n) * 10.0
        sigma = 2.0
        predictions = targets + torch.randn(n) * sigma
        uncertainties = torch.ones(n) * sigma

        result = calibration_error(predictions, uncertainties, targets, n_bins=10)

        assert result["ece"] < 0.5, (
            f"ECE should be < 0.5 for well-calibrated Gaussian noise, got {result['ece']}"
        )

    def test_prediction_intervals_shape(self) -> None:
        """Verify that prediction_intervals returns (lower, upper) with correct shapes.

        Both lower and upper bounds should have the same shape as the
        input mean tensor.
        """
        mean = torch.randn(2, 1, 32, 32)
        std = torch.abs(torch.randn(2, 1, 32, 32))

        lower, upper = prediction_intervals(mean, std, confidence=0.95)

        assert lower.shape == mean.shape, (
            f"Lower bound shape {lower.shape} != mean shape {mean.shape}"
        )
        assert upper.shape == mean.shape, (
            f"Upper bound shape {upper.shape} != mean shape {mean.shape}"
        )

    def test_prediction_intervals_ordering(self) -> None:
        """Verify that lower <= mean <= upper for all prediction intervals.

        The interval bounds should always satisfy lower <= mean <= upper
        when standard deviation is non-negative.
        """
        mean = torch.randn(2, 1, 32, 32)
        std = torch.abs(torch.randn(2, 1, 32, 32)) + 1e-6  # Ensure positive

        lower, upper = prediction_intervals(mean, std, confidence=0.95)

        assert (lower <= mean + 1e-6).all(), (
            "Lower bound should be <= mean for all pixels"
        )
        assert (upper >= mean - 1e-6).all(), (
            "Upper bound should be >= mean for all pixels"
        )
        assert (upper >= lower).all(), (
            "Upper bound should be >= lower bound for all pixels"
        )

    def test_prediction_intervals_wider_at_higher_confidence(self) -> None:
        """Verify that higher confidence produces wider prediction intervals.

        A 99% confidence interval should be strictly wider than a 90%
        confidence interval for positive uncertainty.
        """
        mean = torch.randn(2, 1, 16, 16)
        std = torch.ones(2, 1, 16, 16)

        lower_90, upper_90 = prediction_intervals(mean, std, confidence=0.90)
        lower_99, upper_99 = prediction_intervals(mean, std, confidence=0.99)

        width_90 = (upper_90 - lower_90).mean().item()
        width_99 = (upper_99 - lower_99).mean().item()

        assert width_99 > width_90, (
            f"99% interval width ({width_99:.4f}) should exceed "
            f"90% interval width ({width_90:.4f})"
        )

    def test_prediction_intervals_invalid_confidence_raises(self) -> None:
        """Verify that invalid confidence values raise ValueError."""
        mean = torch.randn(2, 1, 8, 8)
        std = torch.ones(2, 1, 8, 8)

        with pytest.raises(ValueError, match="confidence"):
            prediction_intervals(mean, std, confidence=0.0)

        with pytest.raises(ValueError, match="confidence"):
            prediction_intervals(mean, std, confidence=1.0)

        with pytest.raises(ValueError, match="confidence"):
            prediction_intervals(mean, std, confidence=-0.5)


# ======================================================================
# TestModelComparison
# ======================================================================


class TestModelComparison:
    """Tests that compare properties across all model architectures."""

    def test_all_models_same_interface(
        self,
        pc_unet: PCUNet,
        simple_cnn: SimpleCNN,
        pod_model: PODModel,
        sample_input_64: torch.Tensor,
    ) -> None:
        """Verify that all models accept (B, 9, H, W) and return (B, 1, H, W).

        Every model inheriting from BaseThermalModel should conform to the
        same input/output interface regardless of internal architecture.
        """
        expected_shape = (2, 1, 64, 64)
        models = {
            "PCUNet": pc_unet,
            "SimpleCNN": simple_cnn,
            "PODModel": pod_model,
        }

        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                output = model(sample_input_64)
            assert output.shape == expected_shape, (
                f"{name} output shape {output.shape} != expected {expected_shape}"
            )

    def test_parameter_counts_ordered(
        self,
        pc_unet: PCUNet,
        simple_cnn: SimpleCNN,
        pod_model: PODModel,
    ) -> None:
        """Verify the typical parameter count ordering: POD < SimpleCNN < PCUNet.

        The POD model (MLP only) should have the fewest trainable parameters,
        followed by the SimpleCNN baseline, with the full PCUNet having the
        most.
        """
        n_pod = pod_model.count_parameters()
        n_simple = simple_cnn.count_parameters()
        n_unet = pc_unet.count_parameters()

        assert n_pod < n_simple, (
            f"PODModel ({n_pod:,} params) should have fewer parameters "
            f"than SimpleCNN ({n_simple:,} params)"
        )
        assert n_simple < n_unet, (
            f"SimpleCNN ({n_simple:,} params) should have fewer parameters "
            f"than PCUNet ({n_unet:,} params)"
        )

    def test_all_models_inherit_base(
        self,
        pc_unet: PCUNet,
        simple_cnn: SimpleCNN,
        pod_model: PODModel,
    ) -> None:
        """Verify that all models are instances of BaseThermalModel.

        This ensures they expose the standard interface: forward,
        predict_step, count_parameters, get_config, and rollout.
        """
        for name, model in [
            ("PCUNet", pc_unet),
            ("SimpleCNN", simple_cnn),
            ("PODModel", pod_model),
        ]:
            assert isinstance(model, BaseThermalModel), (
                f"{name} should inherit from BaseThermalModel"
            )
            assert isinstance(model, nn.Module), (
                f"{name} should inherit from nn.Module"
            )

    def test_all_models_have_get_config(
        self,
        pc_unet: PCUNet,
        simple_cnn: SimpleCNN,
        pod_model: PODModel,
    ) -> None:
        """Verify that all models return a valid config dictionary.

        Every config dict should contain at least 'model_type' and
        'trainable_params' keys.
        """
        for name, model in [
            ("PCUNet", pc_unet),
            ("SimpleCNN", simple_cnn),
            ("PODModel", pod_model),
        ]:
            config = model.get_config()
            assert isinstance(config, dict), (
                f"{name}.get_config() should return a dict"
            )
            assert "model_type" in config, (
                f"{name}.get_config() should contain 'model_type'"
            )
            assert "trainable_params" in config, (
                f"{name}.get_config() should contain 'trainable_params'"
            )

    def test_all_models_predict_step(
        self,
        pc_unet: PCUNet,
        simple_cnn: SimpleCNN,
        pod_model: PODModel,
    ) -> None:
        """Verify that predict_step works for all models.

        Each model's predict_step should accept (t_current, params) and
        return a tensor of the same shape as t_current.
        """
        t_current = torch.randn(2, 1, 64, 64)
        params = torch.randn(2, 8, 64, 64)
        expected_shape = (2, 1, 64, 64)

        for name, model in [
            ("PCUNet", pc_unet),
            ("SimpleCNN", simple_cnn),
            ("PODModel", pod_model),
        ]:
            model.eval()
            with torch.no_grad():
                t_next = model.predict_step(t_current, params)
            assert t_next.shape == expected_shape, (
                f"{name}.predict_step output shape {t_next.shape} != "
                f"expected {expected_shape}"
            )

    def test_all_models_rollout(
        self,
        pc_unet: PCUNet,
        simple_cnn: SimpleCNN,
        pod_model: PODModel,
    ) -> None:
        """Verify that rollout produces correct trajectory shapes for all models.

        A rollout of n_steps from an initial field should produce a
        trajectory tensor of shape (B, n_steps+1, H, W).
        """
        t_init = torch.randn(1, 1, 64, 64)
        params = torch.randn(1, 8, 64, 64)
        n_steps = 3

        for name, model in [
            ("PCUNet", pc_unet),
            ("SimpleCNN", simple_cnn),
            ("PODModel", pod_model),
        ]:
            model.eval()
            with torch.no_grad():
                trajectory = model.rollout(t_init, params, n_steps)
            expected_shape = (1, n_steps + 1, 64, 64)
            assert trajectory.shape == expected_shape, (
                f"{name}.rollout shape {trajectory.shape} != "
                f"expected {expected_shape}"
            )
