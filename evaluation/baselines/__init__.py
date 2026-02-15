"""Baseline models for comparison."""

from .fno import FNO2d
from .deeponet import DeepONet
from .simple_cnn import SimpleCNN

__all__ = ['FNO2d', 'DeepONet', 'SimpleCNN']
