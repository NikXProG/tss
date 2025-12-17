import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp
from typing import Callable, Tuple
import logging

logger = logging.getLogger(__name__)


class BoundaryValueProblem:
    """Flexible boundary value problem solver using scipy's solve_bvp."""

    def __init__(self, ode_system: Callable = None, boundary_conditions: Callable = None,
                 x_span: Tuple[float, float] = (0.5, 1.0)):
        if x_span[0] >= x_span[1]:
            raise ValueError("x_span[0] must be less than x_span[1]")
        self.x_span = x_span
        
        if ode_system is not None and not callable(ode_system):
            raise ValueError("ode_system must be callable")
        self.ode_system = ode_system or self.default_ode_system
        
        if boundary_conditions is not None and not callable(boundary_conditions):
            raise ValueError("boundary_conditions must be callable")
        self.boundary_conditions = boundary_conditions or self.default_boundary_conditions
        
        logger.info("BoundaryValueProblem initialized")

    @staticmethod
    def default_ode_system(x, y):
        """Default ODE: y'' + x²y' + (2y)/x² = 1 + 4/x²"""
        dydx = np.zeros_like(y)
        dydx[0] = y[1]
        dydx[1] = x**2 * y[1] + (2 * y[0]) / x**2 + 1 + 4/x**2
        return dydx

    @staticmethod
    def default_boundary_conditions(ya, yb):
        """Default BC: 2y(0.5) - y'(0.5) = 6, y(1) + 3y'(1) = -1"""
        bc = np.zeros(2)
        bc[0] = 2 * ya[0] - ya[1] - 6  # 2y(0.5) - y'(0.5) - 6 = 0
        bc[1] = yb[0] + 3 * yb[1] + 1   # y(1) + 3y'(1) + 1 = 0
        return bc

    def solve(self, n_points: int = 100, tol: float = 1e-8, max_nodes: int = 5000) -> object:
        if n_points < 2:
            raise ValueError("n_points must be at least 2")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        
        x = np.linspace(self.x_span[0], self.x_span[1], n_points)
        y_init = np.zeros((2, x.size))
        y_init[0] = 1.0
        y_init[1] = 0.0
        sol = solve_bvp(
            self.ode_system,
            self.boundary_conditions,
            x,
            y_init,
            tol=tol,
            max_nodes=max_nodes
        )

        if not sol.success:
            logger.warning("BVP solver did not converge completely")
        else:
            logger.info("BVP solved successfully")
        
        return sol

        return sol

    def plot_solution(self, sol, method_name: str = "Решатель КЗ",
                     figsize: Tuple[int, int] = (14, 5), show_plot: bool = True):
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8')
        x_fine = np.linspace(self.x_span[0], self.x_span[1], 200)
        y_fine = sol.sol(x_fine)[0]
        yp_fine = sol.sol(x_fine)[1]

        if not show_plot:
            return x_fine, y_fine, yp_fine

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='white')

        # Первый график: Решение
        ax1.plot(x_fine, y_fine, color='#1f77b4', linewidth=2.5, label='y(x)')
        ax1.set_xlabel('x', fontsize=14, fontweight='bold')
        ax1.set_ylabel('y(x)', fontsize=14, fontweight='bold')
        ax1.set_title('Решение', fontsize=16, fontweight='bold', color='#333333')
        ax1.grid(True, alpha=0.4, linestyle='--')
        ax1.legend(fontsize=12, loc='best')

        # Маркеры граничных точек
        ax1.scatter(self.x_span[0], sol.sol(self.x_span[0])[0], color='#d62728', s=100, marker='o', edgecolor='black', linewidth=1.5, label=f'x={self.x_span[0]:.1f}')
        ax1.scatter(self.x_span[1], sol.sol(self.x_span[1])[0], color='#2ca02c', s=100, marker='o', edgecolor='black', linewidth=1.5, label=f'x={self.x_span[1]:.1f}')

        # Текстовые аннотации
        ax1.annotate(f'y({self.x_span[0]:.1f}) = {sol.sol(self.x_span[0])[0]:.4f}',
                     xy=(self.x_span[0], sol.sol(self.x_span[0])[0]), xytext=(self.x_span[0]+0.05, sol.sol(self.x_span[0])[0]+0.1),
                     fontsize=11, ha='left', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        ax1.annotate(f'y({self.x_span[1]:.1f}) = {sol.sol(self.x_span[1])[0]:.4f}',
                     xy=(self.x_span[1], sol.sol(self.x_span[1])[0]), xytext=(self.x_span[1]-0.05, sol.sol(self.x_span[1])[0]-0.1),
                     fontsize=11, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # Второй график: Решение и производная
        ax2.plot(x_fine, y_fine, color='#1f77b4', linewidth=2.5, label='y(x)')
        ax2.plot(x_fine, yp_fine, color='#ff7f0e', linewidth=2.5, linestyle='--', label="y'(x)")
        ax2.set_xlabel('x', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Значение функции', fontsize=14, fontweight='bold')
        ax2.set_title('Решение и его производная', fontsize=16, fontweight='bold', color='#333333')
        ax2.grid(True, alpha=0.4, linestyle='--')
        ax2.legend(fontsize=12, loc='best')

        plt.suptitle(f'{method_name}: Решение краевой задачи', fontsize=18, fontweight='bold', color='#333333')
        plt.tight_layout()
        plt.show()

        return x_fine, y_fine, yp_fine

    def print_table(self, sol, num_points: int = 11):
        """Print tabular values of the solution"""
        print("\n" + "="*60)
        print("ТАБЛИЧНЫЕ ЗНАЧЕНИЯ РЕШЕНИЯ")
        print("="*60)
        print(f"{'x':^10} {'y(x)':^15} {'y(x)':^15} {'Невязка':^15}")
        print("-"*60)

        x_vals = np.linspace(self.x_span[0], self.x_span[1], num_points)

        for xi in x_vals:
            yi = sol.sol(xi)[0]
            ypi = sol.sol(xi)[1]
            h = 0.001
            ypp = (sol.sol(xi + h)[1] - sol.sol(xi - h)[1]) / (2*h)
            residual = ypp - xi**2 * ypi - (2*yi)/xi**2 - 1 - 4/xi**2
            print(f"{xi:^10.4f} {yi:^15.6f} {ypi:^15.6f} {residual:^15.2e}")

        print("="*60)

        print("\nПРОВЕРКА ГРАНИЧНЫХ УСЛОВИЙ:")
        print("-"*40)

        y_at_a = sol.sol(self.x_span[0])
        bc1 = 2*y_at_a[0] - y_at_a[1]
        print(f"2y({self.x_span[0]:.1f}) - y'({self.x_span[0]:.1f}) = {bc1:.8f} (должно быть 6)")
        print(f"Ошибка: {abs(bc1 - 6):.2e}")

        y_at_b = sol.sol(self.x_span[1])
        bc2 = y_at_b[0] + 3*y_at_b[1]
        print(f"y({self.x_span[1]:.1f}) + 3y'({self.x_span[1]:.1f}) = {bc2:.8f} (должно быть -1)")
        print(f"Ошибка: {abs(bc2 + 1):.2e}")

    def set_custom_problem(self, ode_func: Callable, bc_func: Callable, x_span: Tuple[float, float] = None):
        self.ode_system = ode_func
        self.boundary_conditions = bc_func
        if x_span:
            self.x_span = x_span
