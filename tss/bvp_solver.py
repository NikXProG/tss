import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.optimize import fsolve

class BoundaryValueProblem:
    def __init__(self):
        pass
    
    def ode_system(self, x, y):
        dydx = np.zeros_like(y)
        dydx[0] = y[1]  
        dydx[1] = x**2 * y[1] + (2 * y[0]) / x**2 + 1 + 4/x**2
        
        return dydx
    
    def boundary_conditions(self, ya, yb):
       
        bc = np.zeros(2)
        bc[0] = 2 * ya[0] - ya[1] - 6  # 2y(0.5) - y'(0.5) - 6 = 0
        bc[1] = yb[0] + 3 * yb[1] + 1   # y(1) + 3y'(1) + 1 = 0
        return bc
    
    def solve_bvp(self, n_points=100):
        # Define x grid from 0.5 to 1
        x = np.linspace(0.5, 1, n_points)
        
        # Initial guess for solution [y(x), y'(x)]
        y_init = np.zeros((2, x.size))
        y_init[0] = 1.0  # Guess for y(x)
        y_init[1] = 0.0  # Guess for y'(x)
        
        # Solve BVP
        sol = solve_bvp(self.ode_system, self.boundary_conditions, x, y_init,
                       tol=1e-8, max_nodes=5000)
        
        if not sol.success:
            print("Warning: BVP solver did not converge completely")
        
        return sol
    
    
    def plot_solution(self, sol, method_name="BVP Solver"):
        """Plot the solution and its derivative"""
        x_fine = np.linspace(0.5, 1, 200)
        y_fine = sol.sol(x_fine)[0]
        yp_fine = sol.sol(x_fine)[1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot y(x)
        ax1.plot(x_fine, y_fine, 'b-', linewidth=2, label='y(x)')
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y(x)', fontsize=12)
        ax1.set_title(f'Solution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # Mark boundary points
        ax1.plot(0.5, sol.sol(0.5)[0], 'ro', markersize=8, label='x=0.5')
        ax1.plot(1.0, sol.sol(1.0)[0], 'go', markersize=8, label='x=1.0')
        
        # Add boundary condition values
        ax1.text(0.5, sol.sol(0.5)[0], f'  y(0.5)={sol.sol(0.5)[0]:.4f}', 
                verticalalignment='bottom', fontsize=10)
        ax1.text(1.0, sol.sol(1.0)[0], f'  y(1)={sol.sol(1.0)[0]:.4f}', 
                verticalalignment='top', fontsize=10)
        
        # Plot y(x) and y'(x)
        ax2.plot(x_fine, y_fine, 'b-', linewidth=2, label='y(x)')
        ax2.plot(x_fine, yp_fine, 'r--', linewidth=2, label="y'(x)")
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('Function value', fontsize=12)
        ax2.set_title('Solution and its Derivative', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        plt.show()
        
        return x_fine, y_fine, yp_fine
    
    def print_table(self, x, y, yp, num_points=11):
        """Print tabular values of the solution"""
        print("\n" + "="*60)
        print("TABULAR VALUES OF THE SOLUTION")
        print("="*60)
        print(f"{'x':^10} {'y(x)':^15} {'y(x)':^15} {'Residual':^15}")
        print("-"*60)
        
        indices = np.linspace(0, len(x)-1, num_points, dtype=int)
        
        for idx in indices:
            xi = x[idx]
            yi = y[idx]
            ypi = yp[idx]
            
            if idx > 0 and idx < len(x)-1:
                h = x[idx+1] - x[idx]
                ypp = (yp[idx+1] - yp[idx-1]) / (2*h)
            else:
                ypp = 0  # At boundaries
                
            residual = ypp - xi**2 * ypi - (2*yi)/xi**2 - 1 - 4/xi**2
            
            print(f"{xi:^10.4f} {yi:^15.6f} {ypi:^15.6f} {residual:^15.2e}")
        
        print("="*60)
        
        print("\nBOUNDARY CONDITIONS VERIFICATION:")
        print("-"*40)
        
        y_at_05 = sol.sol(0.5)
        bc1 = 2*y_at_05[0] - y_at_05[1]
        print(f"2y(0.5) - y'(0.5) = {bc1:.8f} (should be 6)")
        print(f"Error: {abs(bc1 - 6):.2e}")
        
        y_at_1 = sol.sol(1.0)
        bc2 = y_at_1[0] + 3*y_at_1[1]
        print(f"y(1) + 3y'(1) = {bc2:.8f} (should be -1)")
        print(f"Error: {abs(bc2 + 1):.2e}")