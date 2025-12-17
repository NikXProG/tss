from typing import List, Tuple
import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class TridiagonalSolver:
    """Class for solving tridiagonal systems using the Thomas algorithm with flexible configuration"""

    def __init__(
        self,
        a: List[float] = None,
        b: List[float] = None,
        c: List[float] = None,
        d: List[float] = None,
    ):
        if a is not None and b is not None and c is not None and d is not None:
            self.set_system(a, b, c, d)
        else:
            self.a = None
            self.b = None
            self.c = None
            self.d = None
            self.n = 0

    def set_system(self, a: List[float], b: List[float], c: List[float], d: List[float]):
        """Set or update the tridiagonal system with validation.

        Args:
            a: Lower diagonal coefficients (n-1 elements)
            b: Main diagonal coefficients (n elements)
            c: Upper diagonal coefficients (n-1 elements)
            d: Right-hand side vector (n elements)

        Raises:
            ValueError: If input lists have incompatible lengths or contain non-numeric values.
        """
        if not all(isinstance(lst, (list, tuple, np.ndarray)) for lst in [a, b, c, d]):
            raise ValueError("All inputs must be lists, tuples, or numpy arrays")
        
        n = len(b)
        if len(a) != n - 1 or len(c) != n - 1 or len(d) != n:
            raise ValueError(f"Incompatible dimensions: a should have {n-1} elements, c should have {n-1}, d should have {n}")
        
        try:
            self.a = np.array(a, dtype=float)
            self.b = np.array(b, dtype=float)
            self.c = np.array(c, dtype=float)
            self.d = np.array(d, dtype=float)
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert inputs to float arrays: {e}")
            raise ValueError(f"Non-numeric values in input: {e}")
        
        self.n = n
        logger.info(f"Tridiagonal system set with {n} equations")

        # Validate dimensions
        if len(a) != self.n - 1 or len(c) != self.n - 1 or len(d) != self.n:
            raise ValueError("Invalid input array dimensions")

    def solve(self) -> np.ndarray:
        """Thomas algorithm (tridiagonal matrix algorithm)"""
        if self.a is None:
            raise ValueError("System not set. Use set_system() first.")

        alpha = np.zeros(self.n)
        beta = np.zeros(self.n)

        alpha[0] = -self.c[0] / self.b[0]
        beta[0] = self.d[0] / self.b[0]

        for i in range(1, self.n - 1):
            denominator = self.b[i] + self.a[i - 1] * alpha[i - 1]
            alpha[i] = -self.c[i] / denominator
            beta[i] = (self.d[i] - self.a[i - 1] * beta[i - 1]) / denominator

        denominator = self.b[self.n - 1] + self.a[self.n - 2] * alpha[self.n - 2]
        beta[self.n - 1] = (
            self.d[self.n - 1] - self.a[self.n - 2] * beta[self.n - 2]
        ) / denominator

        x = np.zeros(self.n)
        x[self.n - 1] = beta[self.n - 1]

        for i in range(self.n - 2, -1, -1):
            x[i] = alpha[i] * x[i + 1] + beta[i]

        return x

    def residual(self, x: np.ndarray = None) -> np.ndarray:
        """Compute residual vector"""
        if x is None:
            x = self.solve()

        residual_vec = np.zeros(self.n)

        residual_vec[0] = self.b[0] * x[0] + self.c[0] * x[1] - self.d[0]

        for i in range(1, self.n - 1):
            residual_vec[i] = (
                self.a[i - 1] * x[i - 1]
                + self.b[i] * x[i]
                + self.c[i] * x[i + 1]
                - self.d[i]
            )

        residual_vec[self.n - 1] = (
            self.a[self.n - 2] * x[self.n - 2]
            + self.b[self.n - 1] * x[self.n - 1]
            - self.d[self.n - 1]
        )

        return residual_vec

    def get_matrix_condition(self) -> float:
        """Estimate condition number of the tridiagonal matrix"""
        if self.a is None:
            return float('inf')

        dominant = True
        for i in range(self.n):
            row_sum = abs(self.b[i])
            if i > 0:
                row_sum += abs(self.a[i-1])
            if i < self.n - 1:
                row_sum += abs(self.c[i])
            if abs(self.b[i]) < row_sum - abs(self.b[i]):
                dominant = False
                break

        if dominant:
            return 1.0
        else:
            return float('inf')

    def plot_solution(self, x: np.ndarray = None, residuals: np.ndarray = None,
                     solution_method: str = "Алгоритм Томаса",
                     figsize: Tuple[int, int] = (12, 5), show_plot: bool = True):
        """Visualize solution and residuals"""
        if x is None:
            x = self.solve()
        if residuals is None:
            residuals = self.residual(x)

        if not show_plot:
            return x, residuals

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='white')

        # 1. Solution plot
        ax1.plot(x, "o-", linewidth=2.5, markersize=8, color="#1f77b4", markerfacecolor="#ff7f0e", markeredgecolor="black", markeredgewidth=1.5)
        ax1.set_xlabel("Индекс i", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Значение x[i]", fontsize=14, fontweight='bold')
        ax1.set_title("Решение системы", fontsize=16, fontweight="bold", color='#333333')
        ax1.grid(True, alpha=0.4, linestyle='--')
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

        # 2. Residuals plot
        ax2.bar(
            range(len(residuals)),
            residuals,
            color="#d62728",
            edgecolor="black",
            alpha=0.8,
            width=0.8
        )
        ax2.set_xlabel("Индекс уравнения", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Невязка", fontsize=14, fontweight='bold')
        ax2.set_title("Невязки решения", fontsize=16, fontweight="bold", color='#333333')
        ax2.grid(True, alpha=0.4, linestyle='--', axis="y")
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

        plt.suptitle(
            f"Решение линейной системы: {solution_method}", fontsize=18, fontweight="bold", color='#333333'
        )
        plt.tight_layout()
        plt.show()

        return x, residuals

    def generate_random_system(self, n: int, diagonal_range: Tuple[float, float] = (1.0, 10.0),
                              off_diagonal_range: Tuple[float, float] = (-2.0, 2.0),
                              rhs_range: Tuple[float, float] = (-10.0, 10.0)) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Generate a random tridiagonal system for testing"""
        np.random.seed(42)  # For reproducibility

        a = [np.random.uniform(*off_diagonal_range) for _ in range(n-1)]
        b = [np.random.uniform(*diagonal_range) for _ in range(n)]
        c = [np.random.uniform(*off_diagonal_range) for _ in range(n-1)]
        d = [np.random.uniform(*rhs_range) for _ in range(n)]

        return a, b, c, d


def solve_variable_coefficients_system(n: int = 10, plot: bool = True):
    """Solve a system with variable coefficients"""

    print("\n" + "=" * 60)
    print(f"Variable Coefficients System ({n}×{n})")
    print("=" * 60)

    solver = TridiagonalSolver()

    a = [1.5] * (n - 1)  # lower diagonal
    b = [4.0 + 0.1 * i for i in range(n)]  # main diagonal
    c = [2.0] * (n - 1)  # upper diagonal
    d = [10.0 + i for i in range(n)]  # right-hand side

    solver.set_system(a, b, c, d)

    print("\nSystem coefficients:")
    print(f"Lower diagonal a: {a}")
    print(f"Main diagonal b: {b}")
    print(f"Upper diagonal c: {c}")
    print(f"Right-hand side d: {d}")
    print()

    solution = solver.solve()

    print("Solution values:")
    for i, val in enumerate(solution):
        print(f"  x[{i}] = {val:.8f}")

    residuals = solver.residual(solution)

    print("\nResidual statistics:")
    print(f"  Maximum residual: {np.max(np.abs(residuals)):.2e}")
    print(f"  Mean residual: {np.mean(np.abs(residuals)):.2e}")

    if plot:
        solver.plot_solution(solution, residuals, f"Алгоритм Томаса ({n}×{n})")

    return solution, residuals
