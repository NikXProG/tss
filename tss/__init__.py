"""TSS Package

A package for solving tridiagonal systems and generating grid functions.
"""

from .grid_function_gen import GridFunctionGenerator
from .thomas_algorithm import (
    TridiagonalSolver,
    plot_solution,
    solve_variable_coefficients_system,
)

__version__ = "0.1.0"
__all__ = [
    "TridiagonalSolver",
    "plot_solution",
    "solve_variable_coefficients_system",
    "GridFunctionGenerator",
]
