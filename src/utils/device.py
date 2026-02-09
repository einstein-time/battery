"""Auto-detection of compute device (CUDA/MPS/CPU) with fallbacks.

Provides cross-platform device selection with graceful degradation
from GPU to CPU when GPU is unavailable or unstable.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Auto-detect the best available compute device.

    Args:
        preferred: Optional preferred device string ('cuda', 'mps', 'cpu').
            If specified and available, this device is used.

    Returns:
        torch.device: The selected compute device.
    """
    if preferred is not None:
        device = torch.device(preferred)
        logger.info("Using preferred device: %s", device)
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info("Using CUDA device: %s (%.1f GB)", gpu_name, gpu_mem)
        return device

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            test_tensor = torch.ones(1, device="mps")
            _ = test_tensor + test_tensor
            del test_tensor
            device = torch.device("mps")
            logger.info("Using MPS device (Apple Silicon)")
            return device
        except Exception as e:
            logger.warning("MPS available but unstable (%s), falling back to CPU", e)

    device = torch.device("cpu")
    logger.info("Using CPU device")
    return device


class DeviceManager:
    """Manages device selection and tensor movement with OOM fallback.

    Attributes:
        device: The active compute device.
    """

    def __init__(self, preferred: Optional[str] = None) -> None:
        """Initialize DeviceManager.

        Args:
            preferred: Optional preferred device string.
        """
        self.device = get_device(preferred)

    def move(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the managed device with OOM fallback.

        Args:
            tensor: Tensor to move.

        Returns:
            Tensor on the managed device, or CPU if OOM occurs.
        """
        try:
            return tensor.to(self.device)
        except RuntimeError as e:
            if "out of memory" in str(e) and self.device.type != "cpu":
                logger.warning("OOM on %s, falling back to CPU", self.device)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                return tensor.to(self.device)
            raise

    def empty_cache(self) -> None:
        """Clear GPU memory cache if applicable."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @property
    def is_gpu(self) -> bool:
        """Check if device is a GPU (CUDA or MPS)."""
        return self.device.type in ("cuda", "mps")
