import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class DampedSpringMass:
    def __init__(self, m=1.0, k=2.0, h=0.5):
        """
        Parameters:
        m: mass (kg)
        k: spring constant (N/m)
        h: damping coefficient (Ns/m)
        """
        self.m = m
        self.k = k
        self.h = h
        
    def forcing_function(self, t, case, b=1.0, w=2.0):
        """Define external forcing functions"""
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
    
    def analytical_solution_case1(self, t, v0):
        """Analytical solution for f=0 (free vibration)"""
        discriminant = self.h**2 - 4*self.m*self.k
        
        if discriminant < 0:  # Underdamped (h^2 < 4km)
            omega_d = np.sqrt(4*self.m*self.k - self.h**2) / (2*self.m)
            return (v0/omega_d) * np.exp(-self.h*t/(2*self.m)) * np.sin(omega_d*t)
        
        elif discriminant > 0:  # Overdamped (h^2 > 4km)
            s1 = (-self.h + np.sqrt(discriminant)) / (2*self.m)
            s2 = (-self.h - np.sqrt(discriminant)) / (2*self.m)
            return v0/(s1 - s2) * (np.exp(s1*t) - np.exp(s2*t))
        
        else:  # Critically damped
            return v0 * t * np.exp(-self.h*t/(2*self.m))
    
    def system_ode(self, t, y, case, b=1.0, w=2.0):
        """ODE system for numerical solution: y = [position, velocity]"""
        x, v = y
        f_ext = self.forcing_function(t, case, b, w)
        
        # dx/dt = v
        # dv/dt = (f_ext - k*x - h*v)/m
        dxdt = v
        dvdt = (f_ext - self.k*x - self.h*v) / self.m
        
        return [dxdt, dvdt]
    
    def solve_numerical(self, t_span, v0, case, b=1.0, w=2.0, dt=0.01):
        """Solve numerically using RK45"""
        y0 = [0.0, v0]  # Initial conditions: x(0)=0, v(0)=v0
        t_eval = np.arange(t_span[0], t_span[1], dt)
        
        sol = solve_ivp(
            fun=lambda t, y: self.system_ode(t, y, case, b, w),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        return sol.t, sol.y[0], sol.y[1]
    
    def plot_solutions(self, t_span=(0, 20), v0=1.0, cases=[1, 2, 3, 4]):
        """Plot analytical and numerical solutions for different cases"""
        # Parameters for overdamped and underdamped cases
        params = [
            {'h': 0.5, 'label': f'Underdamped (h²={0.5**2:.2f} < 4km={4*self.m*self.k:.2f})'},
            {'h': 3.0, 'label': f'Overdamped (h²={3.0**2:.2f} > 4km={4*self.m*self.k:.2f})'}
        ]
        
        # Forcing function labels
        case_labels = {
            1: 'f = 0',
            2: 'f = t-1 (for t≥1)',
            3: 'f = exp(-t)',
            4: 'f = sin(ωt)'
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, case in enumerate(cases):
            ax = axes[idx]
            
            for param in params:
                # Update damping coefficient
                self.h = param['h']
                
                # Numerical solution
                t_num, x_num, v_num = self.solve_numerical(t_span, v0, case)
                
                # Plot numerical solution
                ax.plot(t_num, x_num, linewidth=2, label=f"Numerical ({param['label']})")
                
                # Analytical solution for case 1 (f=0)
                if case == 1:
                    x_analytical = self.analytical_solution_case1(t_num, v0)
                    ax.plot(t_num, x_analytical, '--', linewidth=2, 
                           label=f"Analytical ({param['label'].split('(')[0]})", alpha=0.8)
            
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Displacement (m)', fontsize=12)
            ax.set_title(f'Case: {case_labels[case]}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
            ax.set_xlim(t_span)
        
        plt.tight_layout()
        plt.show()