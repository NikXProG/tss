
"""
Main entry point for the TSS library demonstrations.
"""

from tss.thomas_algorithm import solve_variable_coefficients_system
from tss.grid_function_gen import main as grid_main
from tss.bvp_solver import BoundaryValueProblem
from tss.spring_mass_damper import DampedSpringMass
from tss.derivative_comparison import plot_derivative_comparison
from tss.euler_system import plot_euler_solutions
import numpy as np


def main():
    """Run all demonstrations."""

    # Thomas Algorithm
    print("1. Thomas algorithm")
    solve_variable_coefficients_system()

    # Grid Function Generator
    print("\n2. Grid func gen")
    print("-" * 30)
    grid_main()

    print("\n3. boundary value problem solver")
    print("-" * 35)
    bvp_solver = BoundaryValueProblem()
    print("y'' - x²y' - (2y)/x² = 1 + 4/x²")
    print("Boundary conditions:")
    print("  2y(0.5) - y'(0.5) = 6")
    print("  y(1) + 3y'(1) = -1")
    sol = bvp_solver.solve_bvp(n_points=500)
    if sol.success:
        print("Success")
    else:
        print("Error occurred")

    # Spring Mass Damper
    print("\n4. spring mass damper system")
    print("-" * 30)
    m = 1.0    # mass
    k = 2.0    # spring constant
    system = DampedSpringMass(m=m, k=k)
    v0 = 1.0   # initial velocity
    t_span = (0, 20)  # time span
    system.plot_solutions(t_span=t_span, v0=v0, cases=[1, 2, 3, 4])
    print("Detailed analysis for sinusoidal forcing:")
    print(f"Natural frequency: ω_n = √(k/m) = {np.sqrt(k/m):.3f} rad/s")

    
    print("\n5. Derivative Comparison")
    print("-" * 25)
    plot_derivative_comparison()
 
    print("\n6. Euler method")
    print("-" * 30)
    plot_euler_solutions()


if __name__ == "__main__":
    main()
