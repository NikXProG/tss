
"""
Main entry point for the TSS library demonstrations.
"""

import numpy as np

from tss.bvp_solver import BoundaryValueProblem
from tss.derivative_comparison import DerivativeComparator
from tss.euler_system import EulerSystem
from tss.grid_function_gen import GridFunctionGenerator
from tss.spring_mass_damper import DampedSpringMass
from tss.thomas_algorithm import solve_variable_coefficients_system


def main():
    """Run all demonstrations with flexible configurations."""

    print("1. Алгоритм Томаса с гибкой конфигурацией")
    solution, residuals = solve_variable_coefficients_system(n=10, plot=True)


    custom_functions = {
        'exp(-x/2)': lambda x: np.exp(-x/2),
        'sin(3x)': lambda x: np.sin(3*x),
        'cos²(5x)': lambda x: np.cos(5*x)**2,
        '∑ x^(2n)/(2n)!': lambda x: sum(x**(2*n) / np.math.factorial(2*n) for n in range(20))
    }
    generator = GridFunctionGenerator(functions=custom_functions, default_h=0.05)

    intervals = [
        (0, np.pi/2, "[0; π/2]"),
        (2, 10, "[2; 10]"),
        (-3, 3, "[-3; 3]")
    ]
    generator.plot_all_functions_single_h(intervals, h=0.05)

    # Boundary Value Problem with flexible setup
    print("\n3. Решатель краевых задач с гибкой конфигурацией")
    print("-" * 60)
    bvp_solver = BoundaryValueProblem()
    print("Решаем: y'' - x²y' - (2y)/x² = 1 + 4/x²")
    print("Граничные условия:")
    print("  2y(0.5) - y'(0.5) = 6")
    print("  y(1) + 3y'(1) = -1")
    sol = bvp_solver.solve(n_points=500)
    if sol.success:
        print("✓ Решение сошлось успешно")
        bvp_solver.print_table(sol)
    else:
        print("✗ Решение не сошлось")

    # Spring Mass Damper with flexible parameters
    m, k = 1.5, 3.0
    system = DampedSpringMass(m=m, k=k)
    v0 = 1.5
    t_span = (0, 25)
    system.plot_solutions(t_span=t_span, v0=v0, cases=[1, 2, 3, 4], h_values=[0.3, 4.0])
  

    # Derivative Comparison with custom functions
    comparator = DerivativeComparator(
        steps=[0.01, 0.005, 0.001],
        intervals=[(0, 1), (2, 15), (-5, 5), (0.1, 2)]
    )
    comparator.plot_comparison()

    # Euler System with flexible ODE problems
    euler_solver = EulerSystem(t_span=(0, 15), h=0.01)  # Finer resolution

    # Custom problems
    problems = [
        (EulerSystem.f_a, 1, "y' = 0.5y, y(0)=1", "a"),
        (EulerSystem.f_b, -2, "y' = 2t + 3y, y(0)=-2", "b"),
        (EulerSystem.f_c, np.array([1, 0]), "x1'=x2, x2'=-x1, x1(0)=1, x2(0)=0", "c"),
        (EulerSystem.f_d, np.array([1, 1]), "x1'=x2, x2'=4x1, x1(0)=1, x2(0)=1", "d")
    ]
    euler_solver.plot_solutions(problems=problems)


if __name__ == "__main__":
    main()
