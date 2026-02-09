"""Uncertainty quantification utilities for thermal surrogate models.

Provides functions for Monte Carlo dropout inference, model ensembling,
calibration assessment, prediction interval construction, and reliability
diagram generation.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Monte Carlo dropout prediction
# ----------------------------------------------------------------------


def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate predictive mean and epistemic uncertainty via MC dropout.

    Runs ``n_samples`` stochastic forward passes through *model* with
    dropout layers active and computes per-pixel statistics.

    Args:
        model: A PyTorch model that contains ``Dropout`` or ``Dropout2d``
            layers.  The model is temporarily set to training mode so that
            dropout is active, then restored to its original mode.
        x: Input tensor of shape ``(B, C, H, W)``.
        n_samples: Number of stochastic forward passes (default 20).

    Returns:
        Tuple ``(mean, std)`` where each tensor has the same shape as a
        single model output (typically ``(B, 1, H, W)``).
    """
    was_training = model.training
    model.train()  # Enable dropout

    predictions: List[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x)
            predictions.append(pred)

    # Restore original mode.
    if not was_training:
        model.eval()

    stacked = torch.stack(predictions, dim=0)  # (n_samples, B, ...)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    return mean, std


# ----------------------------------------------------------------------
# Ensemble prediction
# ----------------------------------------------------------------------


def ensemble_predict(
    models: Sequence[nn.Module],
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the ensemble mean and standard deviation.

    Each model in *models* is run in evaluation mode on the same input
    and the predictions are aggregated.

    Args:
        models: Sequence of PyTorch models (e.g. trained with different
            random seeds).
        x: Input tensor of shape ``(B, C, H, W)``.

    Returns:
        Tuple ``(mean, std)`` where each tensor has the same shape as a
        single model output.

    Raises:
        ValueError: If *models* is empty.
    """
    if len(models) == 0:
        raise ValueError("At least one model is required for ensemble prediction.")

    predictions: List[torch.Tensor] = []
    with torch.no_grad():
        for model in models:
            model.eval()
            pred = model(x)
            predictions.append(pred)

    stacked = torch.stack(predictions, dim=0)  # (n_models, B, ...)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    return mean, std


# ----------------------------------------------------------------------
# Calibration error
# ----------------------------------------------------------------------


def calibration_error(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute the expected calibration error (ECE) for regression.

    For regression calibration we check whether predicted confidence
    intervals contain the true values at the rate implied by the
    predicted uncertainty.  We discretise the range of predicted
    standard deviations into ``n_bins`` equal-width bins and, within
    each bin, measure the fraction of true values falling within one
    standard deviation of the prediction.  A perfectly calibrated
    model would have ~68.3 % coverage in every bin.

    Args:
        predictions: Predicted mean values of shape ``(N,)`` or any shape
            that can be flattened.
        uncertainties: Predicted standard deviations (same shape as
            *predictions*).
        targets: Ground-truth values (same shape as *predictions*).
        n_bins: Number of bins for discretising the uncertainty range
            (default 10).

    Returns:
        Dictionary with keys:
            - ``ece``: Expected calibration error (weighted absolute
              deviation from ideal 68.3 % coverage).
            - ``mean_coverage``: Mean empirical coverage across bins.
            - ``ideal_coverage``: The target coverage (0.6827).
    """
    pred_flat = predictions.detach().cpu().flatten().numpy()
    unc_flat = uncertainties.detach().cpu().flatten().numpy()
    tgt_flat = targets.detach().cpu().flatten().numpy()

    # Sort into bins by predicted uncertainty.
    bin_edges = np.linspace(unc_flat.min(), unc_flat.max() + 1e-8, n_bins + 1)
    ideal_coverage: float = 0.6827  # 1-sigma for Gaussian

    weighted_errors: List[float] = []
    coverages: List[float] = []
    total_count = len(pred_flat)

    for i in range(n_bins):
        mask = (unc_flat >= bin_edges[i]) & (unc_flat < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        errors = np.abs(pred_flat[mask] - tgt_flat[mask])
        coverage = float((errors <= unc_flat[mask]).mean())
        weight = float(mask.sum()) / total_count
        weighted_errors.append(weight * abs(coverage - ideal_coverage))
        coverages.append(coverage)

    ece = float(sum(weighted_errors)) if weighted_errors else 0.0
    mean_coverage = float(np.mean(coverages)) if coverages else 0.0

    return {
        "ece": ece,
        "mean_coverage": mean_coverage,
        "ideal_coverage": ideal_coverage,
    }


# ----------------------------------------------------------------------
# Prediction intervals
# ----------------------------------------------------------------------


def prediction_intervals(
    mean: torch.Tensor,
    std: torch.Tensor,
    confidence: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute symmetric Gaussian prediction intervals.

    Args:
        mean: Predicted mean of shape ``(B, 1, H, W)`` or any shape.
        std: Predicted standard deviation (same shape as *mean*).
        confidence: Desired confidence level (default 0.95).

    Returns:
        Tuple ``(lower, upper)`` of the same shape as *mean*.

    Raises:
        ValueError: If *confidence* is not in the open interval (0, 1).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence must be in (0, 1), got {confidence}."
        )

    # z-score for the two-tailed Gaussian interval.
    # We use the inverse CDF (percent-point function).  For common
    # values we avoid importing scipy by using a small lookup; for
    # arbitrary values we fall back to an analytic rational
    # approximation.
    alpha = 1.0 - confidence
    z = _norm_ppf(1.0 - alpha / 2.0)

    z_tensor = torch.tensor(z, dtype=mean.dtype, device=mean.device)
    lower = mean - z_tensor * std
    upper = mean + z_tensor * std
    return lower, upper


def _norm_ppf(p: float) -> float:
    """Approximate the standard-normal percent-point function.

    Uses the rational approximation from Abramowitz & Stegun (formula
    26.2.23) which is accurate to ~4.5e-4 for 0 < p < 1.

    Args:
        p: Probability in (0, 1).

    Returns:
        Approximate z-score such that ``Phi(z) ~= p``.
    """
    import math

    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}.")

    # Coefficients for the rational approximation.
    a0, a1, a2 = 2.515517, 0.802853, 0.010328
    b1, b2, b3 = 1.432788, 0.189269, 0.001308

    if p > 0.5:
        sign = 1.0
        q = 1.0 - p
    else:
        sign = -1.0
        q = p

    t = math.sqrt(-2.0 * math.log(q))
    z = t - (a0 + a1 * t + a2 * t * t) / (1.0 + b1 * t + b2 * t * t + b3 * t * t * t)
    return sign * z


# ----------------------------------------------------------------------
# Reliability diagram data
# ----------------------------------------------------------------------


def reliability_diagram(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """Generate data for a reliability (calibration) diagram.

    For a set of confidence levels (equally spaced between 0 and 1) the
    function computes the *expected* and *observed* fractions of target
    values falling within the corresponding prediction interval.

    A perfectly calibrated model produces points along the diagonal
    (expected == observed).

    Args:
        predictions: Predicted mean values (any shape, will be flattened).
        uncertainties: Predicted standard deviations (same shape).
        targets: Ground-truth values (same shape).
        n_bins: Number of confidence levels to evaluate (default 10).

    Returns:
        Dictionary with keys:
            - ``expected_coverage``: 1-D array of shape ``(n_bins,)``
              with the nominal confidence levels.
            - ``observed_coverage``: 1-D array of shape ``(n_bins,)``
              with the empirical fraction of targets inside each
              interval.
            - ``bin_counts``: 1-D array of shape ``(n_bins,)`` with the
              total number of pixels evaluated at each level (constant
              here since every pixel is evaluated at every level).
    """
    pred_flat = predictions.detach().cpu().flatten().numpy()
    unc_flat = uncertainties.detach().cpu().flatten().numpy()
    tgt_flat = targets.detach().cpu().flatten().numpy()

    # Confidence levels from ~0 to ~1 (excluding exact 0 and 1 which
    # would correspond to z=0 and z=inf respectively).
    confidence_levels = np.linspace(1.0 / (n_bins + 1), n_bins / (n_bins + 1), n_bins)

    expected_coverage = np.empty(n_bins, dtype=np.float64)
    observed_coverage = np.empty(n_bins, dtype=np.float64)
    bin_counts = np.full(n_bins, fill_value=len(pred_flat), dtype=np.int64)

    abs_errors = np.abs(pred_flat - tgt_flat)

    for i, conf in enumerate(confidence_levels):
        z = _norm_ppf(0.5 + conf / 2.0)  # two-tailed z-score
        radius = z * unc_flat
        fraction_inside = float((abs_errors <= radius).mean())
        expected_coverage[i] = conf
        observed_coverage[i] = fraction_inside

    return {
        "expected_coverage": expected_coverage,
        "observed_coverage": observed_coverage,
        "bin_counts": bin_counts,
    }
