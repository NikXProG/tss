from math import atan, cos, exp, log, sin, sqrt
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


class DerivativeComparator:
    """Flexible class for comparing numerical and analytical derivatives."""

    def __init__(self, functions: List[Callable] = None, derivatives: List[Callable] = None,
                 intervals: List[Tuple[float, float]] = None, steps: List[float] = None,
                 func_names: List[str] = None):
        """Initialize with default or custom functions, derivatives, intervals, steps, and names."""
        if functions is None:
            self.functions = [
                lambda x: exp(-x/2),
                lambda x: (sin(3*x**4/5))**3,
                lambda x: (cos(x/(x+1)))**2,
                lambda x: log(x + sqrt(4 + x**2)),
                lambda x: (x * atan(2*x)) / (x**2 + 4)
            ]
        else:
            self.functions = functions

        if derivatives is None:
            self.derivatives = [
                lambda x: -0.5 * exp(-x/2),
                lambda x: 3*(sin(3*x**4/5))**2 * cos(3*x**4/5) * (12*x**3/5),
                lambda x: -2*cos(x/(x+1))*sin(x/(x+1)) * (1/(x+1)**2),
                lambda x: 1/sqrt(4 + x**2),
                lambda x: (atan(2*x)*(x**2+4) - x*(2*x/(4*x**2+1))) / ((x**2+4)**2)
            ]
        else:
            self.derivatives = derivatives

        if intervals is None:
            self.intervals = [(0, 1), (2, 15), (-5, 5)]
        else:
            self.intervals = intervals

        if steps is None:
            self.steps = [0.01, 0.005]
        else:
            self.steps = steps

        if func_names is None:
            self.func_names = ['exp(-x/2)', 'sin^3(3x^4/5)', 'cos^2(x/(x+1))', 'ln(x+sqrt(4+x^2))', '(x*arctg2x)/(x^2+4)']
        else:
            self.func_names = func_names

        # Validate lengths
        n_funcs = len(self.functions)
        assert len(self.derivatives) == n_funcs, "Derivatives list must match functions"
        assert len(self.func_names) == n_funcs, "Function names must match functions"

    def numerical_derivative(self, f: Callable, a: float, b: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute numerical derivative using forward difference."""
        x_vals = np.arange(a, b + h/2, h)
        y_vals = np.array([f(x) for x in x_vals])
        dydx = np.zeros_like(x_vals)
        dydx[:-1] = (y_vals[1:] - y_vals[:-1]) / h
        dydx[-1] = dydx[-2]  # Extend last value
        return x_vals, dydx

    def plot_comparison(self, figsize: Tuple[int, int] = (18, 22)):
        """Plot comparison of numerical vs analytical derivatives."""
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(len(self.functions), len(self.intervals),
                                figsize=figsize, constrained_layout=True, facecolor='white')

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, (func, deriv, name) in enumerate(zip(self.functions, self.derivatives, self.func_names)):
            for j, (a, b) in enumerate(self.intervals):
                ax = axes[i, j]
                x_fine = np.linspace(a, b, 1000)
                y_analytic = [deriv(x) for x in x_fine]
                ax.plot(x_fine, y_analytic, color='black', linewidth=3, label='Аналитическая')

                for k, h in enumerate(self.steps):
                    x_num, y_num = self.numerical_derivative(func, a, b, h)
                    ax.plot(x_num, y_num, '--', linewidth=2, color=colors[k % len(colors)], label=f'h={h}')

                ax.set_xlabel('x', fontsize=12, fontweight='bold')
                ax.set_ylabel("y'", fontsize=12, fontweight='bold')
                ax.set_title(f'{name}\nИнтервал [{a}, {b}]', fontsize=14, fontweight='bold', color='#333333')
                ax.grid(True, alpha=0.4, linestyle='--')
                if j == len(self.intervals) - 1:
                    ax.legend(loc='best', fontsize=10)

        plt.suptitle('Численное vs Аналитическое Производные', fontsize=18, fontweight='bold', color='#333333', y=1.02)
        plt.show()

    def compute_errors(self, func_idx: int, interval_idx: int, h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute errors between numerical and analytical derivatives for a specific function and interval."""
        func = self.functions[func_idx]
        deriv = self.derivatives[func_idx]
        a, b = self.intervals[interval_idx]

        x_num, y_num = self.numerical_derivative(func, a, b, h)
        y_analytic = np.array([deriv(x) for x in x_num])
        errors = np.abs(y_num - y_analytic)

        return x_num, y_num, y_analytic, errors
