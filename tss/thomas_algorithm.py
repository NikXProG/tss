from typing import List

import matplotlib.pyplot as plt
import numpy as np


class TridiagonalSolver:
    """Class for solving tridiagonal systems using the Thomas algorithm"""

    def __init__(
        self,
        a: List[float],  # lower diagonal (indices 0..n-2)
        b: List[float],  # main diagonal (indices 0..n-1)
        c: List[float],  # upper diagonal (indices 0..n-2)
        d: List[float],  # right-hand side (indices 0..n-1)
    ):
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c = np.array(c, dtype=float)
        self.d = np.array(d, dtype=float)
        self.n = len(b)

        # Validate dimensions
        if len(a) != self.n - 1 or len(c) != self.n - 1 or len(d) != self.n:
            raise ValueError("Invalid input array dimensions")

    def solve(self) -> np.ndarray:
        """Thomas algorithm (tridiagonal matrix algorithm)"""
        # Forward sweep
        alpha = np.zeros(self.n)
        beta = np.zeros(self.n)

        # Initial coefficients
        alpha[0] = -self.c[0] / self.b[0]
        beta[0] = self.d[0] / self.b[0]

        # forward coefficients
        for i in range(1, self.n - 1):
            denominator = self.b[i] + self.a[i - 1] * alpha[i - 1]
            alpha[i] = -self.c[i] / denominator
            beta[i] = (self.d[i] - self.a[i - 1] * beta[i - 1]) / denominator

        # Last beta coefficient
        denominator = self.b[self.n - 1] + self.a[self.n - 2] * alpha[self.n - 2]
        beta[self.n - 1] = (
            self.d[self.n - 1] - self.a[self.n - 2] * beta[self.n - 2]
        ) / denominator

        # Backward substitution
        x = np.zeros(self.n)
        x[self.n - 1] = beta[self.n - 1]

        for i in range(self.n - 2, -1, -1):
            x[i] = alpha[i] * x[i + 1] + beta[i]

        return x

    def residual(self, x: np.ndarray) -> np.ndarray:
        """Compute residual vector"""
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


def plot_solution(
    x: np.ndarray, residuals: np.ndarray, solution_method: str = "Thomas Algorithm"
):
    """Visualize solution and residuals"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Solution plot
    ax1.plot(x, "o-", linewidth=2, markersize=8, color="royalblue")
    ax1.set_xlabel("Index i", fontsize=12)
    ax1.set_ylabel("Value x[i]", fontsize=12)
    ax1.set_title("System Solution", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # 2. Residuals plot
    ax2.bar(
        range(len(residuals)),
        residuals,
        color="lightcoral",
        edgecolor="darkred",
        alpha=0.7,
    )
    ax2.set_xlabel("Equation Index", fontsize=12)
    ax2.set_ylabel("Residual", fontsize=12)
    ax2.set_title("Solution Residuals", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.suptitle(
        f"Linear System Solution: {solution_method}", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


def solve_variable_coefficients_system():
    """Solve a 10×10 system with variable coefficients"""

    print("\n" + "=" * 60)
    print("Variable Coefficients System (10×10)")
    print("=" * 60)

    n = 10

    a = [1.5] * (n - 1)  # lower diagonal
    b = [4.0 + 0.1 * i for i in range(n)]  # main diagonal
    c = [2.0] * (n - 1)  # upper diagonal
    d = [10.0 + i for i in range(n)]  # right-hand side

    solver = TridiagonalSolver(a, b, c, d)

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

    plt.figure(figsize=(10, 6))

    plt.plot(
        solution,
        "o-",
        linewidth=2,
        markersize=8,
        color="royalblue",
        label="Solution x[i]",
    )

    plt.errorbar(
        range(n),
        solution,
        yerr=np.abs(residuals) / 10,
        fmt="none",
        ecolor="red",
        alpha=0.5,
        capsize=3,
        label="Residuals (scaled)",
    )

    plt.xlabel("Index i", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title(
        "Solution of 10×10 Tridiagonal System\nwith Variable Coefficients",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
