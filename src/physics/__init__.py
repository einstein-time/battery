"""Physics solver and validation modules for 2D heat equation."""

from src.physics.solver import HeatSolver2D
from src.physics.materials import MaterialProperties, create_material_mask
from src.physics.validation import manufactured_solution_test

__all__ = [
    "HeatSolver2D",
    "MaterialProperties",
    "create_material_mask",
    "manufactured_solution_test",
]
