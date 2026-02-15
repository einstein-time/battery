"""
Honest error metrics - no DC offset tricks.
"""

import torch
import numpy as np


def compute_metrics(predictions, targets):
    """
    Compute publication-quality error metrics.

    Args:
        predictions: (N, ...) torch.Tensor
        targets: (N, ...) torch.Tensor

    Returns:
        dict with honest metrics
    """
    # Convert to tensors if needed
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Absolute errors
    mae = torch.abs(predictions - targets).mean().item()
    rmse = torch.sqrt(torch.mean((predictions - targets)**2)).item()

    # Temperature range (for normalization)
    T_min = targets.min().item()
    T_max = targets.max().item()
    T_range = T_max - T_min

    # Normalized errors (by RANGE, not magnitude)
    nrmse = rmse / T_range
    nmae = mae / T_range

    # Relative L2 (with context about DC offset)
    rel_l2 = (torch.norm(predictions - targets) / torch.norm(targets)).item()

    # Max error
    max_error = torch.abs(predictions - targets).max().item()

    return {
        'mae_K': mae,
        'rmse_K': rmse,
        'nrmse': nrmse,
        'nmae': nmae,
        'max_error_K': max_error,
        'rel_l2': rel_l2,
        'T_range_K': T_range,
        'T_min_K': T_min,
        'T_max_K': T_max
    }


def print_metrics(metrics, title="Metrics"):
    """Print metrics in publication format."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"\n📏 ABSOLUTE ERRORS (preferred for reporting):")
    print(f"  MAE:        {metrics['mae_K']:.4f} K")
    print(f"  RMSE:       {metrics['rmse_K']:.4f} K")
    print(f"  Max Error:  {metrics['max_error_K']:.4f} K")

    print(f"\n📊 NORMALIZED ERRORS (by temperature range):")
    print(f"  NRMSE:      {metrics['nrmse']:.4f} ({metrics['nrmse']*100:.2f}%)")
    print(f"  NMAE:       {metrics['nmae']:.4f} ({metrics['nmae']*100:.2f}%)")

    print(f"\n⚠️  RELATIVE L2 (context required):")
    print(f"  Rel L2:     {metrics['rel_l2']:.6f} ({metrics['rel_l2']*100:.4f}%)")
    print(f"  ⚠️  WARNING: Large DC offset ({metrics['T_min_K']:.1f}K) deflates this metric!")
    print(f"  Temperature range: {metrics['T_range_K']:.1f} K ({metrics['T_min_K']:.1f} - {metrics['T_max_K']:.1f} K)")

    print(f"\n💡 FOR PUBLICATION: Report MAE, RMSE, NRMSE (NOT relative L2)")
    print(f"{'='*60}\n")
