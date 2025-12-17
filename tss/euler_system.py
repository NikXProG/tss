import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, List, Tuple, Union


class EulerSystem:
    """Flexible Euler integrator for scalar or vector ODE systems."""

    def __init__(self, t_span: Tuple[float, float] = (0.0, 10.0), h: float = 0.025):
        self.t_span = t_span
        self.h = h

    def integrate(self, f: Callable, y0: Union[float, np.ndarray],
                  t_span: Tuple[float, float] = None, h: float = None) -> Tuple[np.ndarray, np.ndarray]:
        if t_span is None:
            t_span = self.t_span
        if h is None:
            h = self.h

        t0, tf = t_span
        n_steps = int((tf - t0) / h)
        t = np.linspace(t0, tf, n_steps + 1)

        is_scalar = np.isscalar(y0)

        if is_scalar:
            y = np.zeros(n_steps + 1)
            y[0] = y0
            for i in range(n_steps):
                y[i + 1] = y[i] + h * f(t[i], y[i])
        else:
            y = np.zeros((n_steps + 1, len(y0)))
            y[0] = y0
            for i in range(n_steps):
                y[i + 1] = y[i] + h * f(t[i], y[i])

        return t, y

    def plot_solutions(self, problems: List[Tuple[Callable, Union[float, np.ndarray], str, str]] = None,
                      t_span: Tuple[float, float] = None, h: float = None,
                      figsize: Tuple[int, int] = (12, 10)):
        """Plot solutions for multiple ODE problems."""
        if problems is None:
            # Default examples
            problems = [
                (self.f_a, 1, "y' = 0.5y, y(0)=1", "a"),
                (self.f_b, -2, "y' = 2t + 3y, y(0)=-2", "b"),
                (self.f_c, np.array([1, 0]), "x1'=x2, x2'=-x1, x1(0)=1, x2(0)=0", "c"),
                (self.f_d, np.array([1, 1]), "x1'=x2, x2'=4x1, x1(0)=1, x2(0)=1", "d")
            ]

        if t_span is None:
            t_span = self.t_span
        if h is None:
            h = self.h

        n_problems = len(problems)
        n_cols = 2
        n_rows = (n_problems + 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, facecolor='white')
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        axes = axes.flatten()

        fig.suptitle(f'Численные решения методом Эйлера (h={h})', fontsize=16, fontweight='bold', color='#333333')

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for idx, (f, y0, title, label) in enumerate(problems):
            ax = axes[idx]
            t, y = self.integrate(f, y0, t_span, h)

            if np.isscalar(y0):
                ax.plot(t, y, color=colors[0], linewidth=2.5)
            else:
                for i in range(y.shape[1]):
                    ax.plot(t, y[:, i], color=colors[i % len(colors)], label=f'x{i+1}(t)', linewidth=2.5)

            ax.set_title(f"{label}) {title}", fontsize=14, fontweight='bold', color='#333333')
            ax.set_xlabel('t', fontsize=12, fontweight='bold')
            ax.set_ylabel('y(t)' if np.isscalar(y0) else 'x(t)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.4, linestyle='--')
            if not np.isscalar(y0):
                ax.legend(fontsize=10)

        # Hide unused subplots
        for idx in range(n_problems, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

    # Static example functions
    @staticmethod
    def f_a(t, y):
        return 0.5 * y

    @staticmethod
    def f_b(t, y):
        return 2 * t + 3 * y

    @staticmethod
    def f_c(t, x):
        x1, x2 = x
        return np.array([x2, -x1])

    @staticmethod
    def f_d(t, x):
        x1, x2 = x
        return np.array([x2, 4 * x1])


# Compatibility wrapper
def plot_euler_solutions(t_span=(0, 10), h=0.025):
    """Compatibility wrapper used by main.py"""
    es = EulerSystem(t_span=t_span, h=h)
    es.plot_solutions()

