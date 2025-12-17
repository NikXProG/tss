"""TSS Package"""

from .bvp_solver import BoundaryValueProblem
from .derivative_comparison import DerivativeComparator
from .euler_system import EulerSystem
from .grid_function_gen import GridFunctionGenerator
from .spring_mass_damper import DampedSpringMass
from .thomas_algorithm import (
    TridiagonalSolver,
    solve_variable_coefficients_system,
)

__version__ = "0.2.0"
__all__ = [
    "BoundaryValueProblem",
    "DerivativeComparator",
    "EulerSystem",
    "GridFunctionGenerator",
    "DampedSpringMass",
    "TridiagonalSolver",
    "solve_variable_coefficients_system",
]
