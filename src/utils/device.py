"""Cross-platform device management for PyTorch.

Automatically detects and selects the best available device:
  - CUDA (NVIDIA GPU) if available
  - MPS (Apple Silicon GPU) if available and stable
  - CPU as fallback

All code is cross-platform compatible (Windows, macOS, Linux, Colab).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def get_device(
    prefer_mps: bool = False,
    verbose: bool = True,
) -> torch.device:
    """Get the best available PyTorch device.

    Priority:
      1. CUDA (NVIDIA GPU) - most stable and performant
      2. MPS (Apple Silicon) - if explicitly requested via prefer_mps=True
      3. CPU - universal fallback

    Args:
        prefer_mps: If True, prefer MPS over CPU on Apple Silicon.
            Default False due to some PyTorch operations not being MPS-compatible.
        verbose: If True, log the selected device.

    Returns:
        Selected torch.device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using CUDA device: {gpu_name}")
    elif prefer_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        if verbose:
            logger.info("Using CPU device")

    return device


def setup_device(
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> torch.device:
    """Setup device and random seeds for reproducibility.

    Args:
        device: Explicit device to use. If None, auto-detect with get_device().
        seed: Random seed for reproducibility.

    Returns:
        Selected torch.device.
    """
    if device is None:
        device = get_device()

    # Set random seeds
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        # Enable cuDNN benchmarking for performance
        torch.backends.cudnn.benchmark = True

    logger.info(f"Device set to: {device}, seed={seed}")

    return device


def get_device_properties() -> dict:
    """Get information about available compute devices.

    Returns:
        Dictionary with device information.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info
