"""Physics-based simulation and validation utilities."""

from src.physics.solver import HeatSolver2D
from src.physics.materials import (
    MaterialProperties,
    create_material_mask,
    compute_signed_distance,
)

__all__ = [
    "HeatSolver2D",
    "MaterialProperties",
    "create_material_mask",
    "compute_signed_distance",
]
