import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, List, Tuple


class DampedSpringMass:
    """Flexible damped spring-mass system simulator."""

    def __init__(self, m: float = 1.0, k: float = 2.0, h: float = 0.5):
        self.m = m
        self.k = k
        self.h = h

    def forcing_function(self, t: float, case: int, b: float = 1.0, w: float = 2.0) -> float:
        if case == 1:
            return 0.0
        elif case == 2:
            return t - 1 if t >= 1 else 0.0
        elif case == 3:
            return np.exp(-t)
        elif case == 4:
            return b * np.sin(w * t)
        else:
            return 0.0

    def analytical_solution_case1(self, t: np.ndarray, v0: float) -> np.ndarray:
        discriminant = self.h**2 - 4*self.m*self.k

        if discriminant < 0:
            omega_d = np.sqrt(4*self.m*self.k - self.h**2) / (2*self.m)
            return (v0/omega_d) * np.exp(-self.h*t/(2*self.m)) * np.sin(omega_d*t)

        elif discriminant > 0:
            s1 = (-self.h + np.sqrt(discriminant)) / (2*self.m)
            s2 = (-self.h - np.sqrt(discriminant)) / (2*self.m)
            return v0/(s1 - s2) * (np.exp(s1*t) - np.exp(s2*t))

        else:
            return v0 * t * np.exp(-self.h*t/(2*self.m))

    def system_ode(self, t: float, y: np.ndarray, case: int, b: float = 1.0, w: float = 2.0) -> np.ndarray:
        x, v = y
        f_ext = self.forcing_function(t, case, b, w)

        dxdt = v
        dvdt = (f_ext - self.k*x - self.h*v) / self.m

        return np.array([dxdt, dvdt])

    def solve_numerical(self, t_span: Tuple[float, float], v0: float, case: int,
                       b: float = 1.0, w: float = 2.0, dt: float = 0.01,
                       method: str = 'RK45', rtol: float = 1e-8, atol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve numerically using scipy's solve_ivp"""
        y0 = np.array([0.0, v0])  # Initial conditions: x(0)=0, v(0)=v0
        t_eval = np.arange(t_span[0], t_span[1], dt)

        sol = solve_ivp(
            fun=lambda t, y: self.system_ode(t, y, case, b, w),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol
        )

        return sol.t, sol.y[0], sol.y[1]

    def plot_solutions(self, t_span: Tuple[float, float] = (0, 20), v0: float = 1.0,
                      cases: List[int] = [1, 2, 3, 4], h_values: List[float] = None,
                      figsize: Tuple[int, int] = (14, 10), show_analytical: bool = True):
        """Plot analytical and numerical solutions for different cases"""
        if h_values is None:
            h_values = [0.5, 3.0]  # Underdamped and overdamped

        # Forcing function labels
        case_labels = {
            1: 'f = 0',
            2: 'f = t-1 (для t≥1)',
            3: 'f = exp(-t)',
            4: 'f = sin(ωt)'
        }

        fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor='white')
        axes = axes.flatten()

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for idx, case in enumerate(cases):
            ax = axes[idx]

            for k, h_val in enumerate(h_values):
                # Temporarily change damping
                original_h = self.h
                self.h = h_val

                discriminant = self.h**2 - 4*self.m*self.k
                if discriminant < 0:
                    damping_type = 'Недозатухающий'
                elif discriminant > 0:
                    damping_type = 'Перезагухающий'
                else:
                    damping_type = 'Критически затухающий'

                # Numerical solution
                t_num, x_num, v_num = self.solve_numerical(t_span, v0, case)

                # Plot numerical solution
                label = f"Численный ({damping_type})"
                ax.plot(t_num, x_num, color=colors[k], linewidth=2.5, label=label)

                # Analytical solution for case 1 (f=0)
                if case == 1 and show_analytical:
                    x_analytical = self.analytical_solution_case1(t_num, v0)
                    ax.plot(t_num, x_analytical, '--', color=colors[k], linewidth=2.5, alpha=0.8,
                           label=f"Аналитический ({damping_type})")

                # Restore original damping
                self.h = original_h

            ax.set_xlabel('Время (с)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Перемещение (м)', fontsize=14, fontweight='bold')
            ax.set_title(f'Случай: {case_labels[case]}', fontsize=16, fontweight='bold', color='#333333')
            ax.grid(True, alpha=0.4, linestyle='--')
            ax.legend(loc='best', fontsize=11)
            ax.set_xlim(t_span)

        plt.suptitle('Решения для пружинно-массовой системы с демпфированием', fontsize=18, fontweight='bold', color='#333333')
        plt.tight_layout()
        plt.show()

    def get_natural_frequency(self) -> float:
        """Calculate natural frequency ω_n = √(k/m)"""
        return np.sqrt(self.k / self.m)

    def get_damping_ratio(self) -> float:
        """Calculate damping ratio ζ = h/(2√(km))"""
        return self.h / (2 * np.sqrt(self.k * self.m))

    def simulate_custom_forcing(self, forcing_func: Callable[[float], float],
                               t_span: Tuple[float, float] = (0, 20), v0: float = 1.0,
                               dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate with custom forcing function"""
        y0 = np.array([0.0, v0])
        t_eval = np.arange(t_span[0], t_span[1], dt)

        def custom_ode(t, y):
            x, v = y
            f_ext = forcing_func(t)
            dxdt = v
            dvdt = (f_ext - self.k*x - self.h*v) / self.m
            return np.array([dxdt, dvdt])

        sol = solve_ivp(
            fun=custom_ode,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )

        return sol.t, sol.y[0], sol.y[1]
