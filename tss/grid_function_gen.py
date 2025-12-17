from math import factorial
from typing import Callable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


class GridFunctionGenerator:
    """Class for generating grid functions"""

    def __init__(self):
        """Initialize functions and their names"""
        self.functions = {
            'exp(-x/2)': lambda x: np.exp(-x/2),
            'sin(3x)': lambda x: np.sin(3*x),
            'cos²(5x)': lambda x: np.cos(5*x)**2,
            '∑ x^(2n)/(2n)!': self._cosh_like_series
        }

        self.function_names = list(self.functions.keys())

    def _cosh_like_series(self, x: Union[float, np.ndarray], n_terms: int = 20) -> Union[float, np.ndarray]:
        """Calculate series sum_{n=0}^∞ x^(2n)/(2n)!"""
        if isinstance(x, np.ndarray):
            result = np.zeros_like(x, dtype=np.float64)
            x = x.astype(np.float64)
            for n in range(n_terms):
                denominator = float(factorial(2*n))
                result += x**(2*n) / denominator
            return result
        else:
            result = 0.0
            for n in range(n_terms):
                denominator = float(factorial(2*n))
                result += x**(2*n) / denominator
            return result

    def generate_grid(self, a: float, b: float, h: float) -> np.ndarray:
        """Generate uniform grid"""
        n_points = int((b - a) / h) + 1
        return np.linspace(a, b, n_points, dtype=np.float64)

    def generate_grid_function(self, func: Callable, a: float, b: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate grid function"""
        x_grid = self.generate_grid(a, b, h)
        y_grid = func(x_grid)
        return x_grid, y_grid

    def create_fine_grid(self, a: float, b: float, n_points: int = 1000) -> np.ndarray:
        """Create fine grid for smooth plots"""
        return np.linspace(a, b, n_points, dtype=np.float64)

    def plot_function_comparison(self, func_name: str, interval: Tuple[float, float],
                                h: float, n_terms: int = 20):
        """Plot exact function and grid approximation for given interval and step size"""
        func = self.functions[func_name]

        if func_name == '∑ x^(2n)/(2n)!':
            def actual_func(x):
                return self._cosh_like_series(x, n_terms)
        else:
            actual_func = func

        a, b = interval
        interval_label = f"[{a:.2f}; {b:.2f}]"

        # Create fine grid for exact function
        x_fine = self.create_fine_grid(a, b, 1000)
        y_exact = actual_func(x_fine)

        # Generate grid function
        x_grid, y_grid = self.generate_grid_function(actual_func, a, b, h)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Exact function
        ax1.plot(x_fine, y_exact, 'b-', linewidth=2, label='Exact function')
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('f(x)', fontsize=12)
        ax1.set_title(f'Exact function: {func_name}', fontsize=14)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend()

        # Plot 2: Grid approximation
        step = max(1, len(x_grid) // 50)  # Show subset of points for clarity
        x_display = x_grid[::step]
        y_display = y_grid[::step]

        ax2.plot(x_fine, y_exact, 'b-', linewidth=1, alpha=0.5, label='Exact function')
        ax2.plot(x_display, y_display, 'ro', markersize=6, markeredgewidth=1,
                markeredgecolor='black', label=f'Grid points (h={h})', alpha=0.8)
        ax2.plot(x_grid, y_grid, 'r-', linewidth=1, alpha=0.3, label='Grid interpolation')

        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('f(x)', fontsize=12)
        ax2.set_title(f'Grid approximation: {func_name} on {interval_label}', fontsize=14)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend()

        plt.suptitle(f'Function: {func_name}, Interval: {interval_label}, Step: h={h}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Print grid information
        print(f"Function: {func_name:<20} Interval: {interval_label:<15} h = {h}")
        print(f"Number of grid points: {len(x_grid)}")
        print(f"y_min = {y_grid.min():.6f}, y_max = {y_grid.max():.6f}, y_mean = {y_grid.mean():.6f}")
        print("-" * 70)

        return x_grid, y_grid, x_fine, y_exact

    def plot_all_functions_single_h(self, intervals: List[Tuple[float, float, str]],
                                   h: float, n_terms: int = 20):
        """Plot all functions on different intervals with the same step size h"""
        n_intervals = len(intervals)
        n_functions = len(self.function_names)

        fig, axes = plt.subplots(n_functions, n_intervals, figsize=(5*n_intervals, 4*n_functions))

        if n_functions == 1:
            axes = axes.reshape(1, -1)
        if n_intervals == 1:
            axes = axes.reshape(-1, 1)

        for i, func_name in enumerate(self.function_names):
            func = self.functions[func_name]

            if func_name == '∑ x^(2n)/(2n)!':
                def actual_func(x):
                    return self._cosh_like_series(x, n_terms)
            else:
                actual_func = func

            for j, (a, b, interval_name) in enumerate(intervals):
                ax = axes[i, j]

                # Create fine grid for exact function
                x_fine = self.create_fine_grid(a, b, 500)
                y_exact = actual_func(x_fine)

                # Generate grid function
                x_grid, y_grid = self.generate_grid_function(actual_func, a, b, h)

                # Show subset of points for clarity
                step = max(1, len(x_grid) // 30)
                x_display = x_grid[::step]
                y_display = y_grid[::step]

                # Plot exact function
                ax.plot(x_fine, y_exact, 'b-', linewidth=2, alpha=0.7, label='Exact')

                # Plot grid points
                ax.plot(x_display, y_display, 'ro', markersize=5, markeredgewidth=1,
                       markeredgecolor='black', label=f'Grid (h={h})', alpha=0.8)

                # Plot grid lines
                ax.plot(x_grid, y_grid, 'r-', linewidth=1, alpha=0.3)

                ax.set_xlabel('x', fontsize=10)
                ax.set_ylabel('f(x)', fontsize=10)
                ax.set_title(f'{func_name}\n{interval_name}', fontsize=11)
                ax.grid(True, alpha=0.3, linestyle=':')
                ax.legend(fontsize=9)

        plt.suptitle(f'Grid Function Analysis (h = {h})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()



def main():
    """Main function"""

    # Initialize generator
    generator = GridFunctionGenerator()

    # Define intervals with names
    intervals = [
        (0, np.pi/2, "[0; π/2]"),
        (2, 10, "[2; 10]"),
        (-3, 3, "[-3; 3]")
    ]

    #h_values = [0.1, 0.05, 0.01]

    default_h = 0.1


    # 1. Plot all functions on all intervals with default step size
    print("\nPLOTTING ALL FUNCTIONS WITH DEFAULT STEP SIZE")
    print("-" * 50)
    generator.plot_all_functions_single_h(intervals, default_h)
