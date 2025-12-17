import numpy as np
import matplotlib.pyplot as plt

def euler_system(f, y0, t_span, h):
    """
    Euler method for system of ODEs
    f: function returning derivatives, f(t, y)
    y0: initial conditions
    t_span: [t0, tf]
    h: step size
    """
    t0, tf = t_span
    n_steps = int((tf - t0) / h)
    t = np.linspace(t0, tf, n_steps + 1)
    
    if isinstance(y0, (int, float)):
        y = np.zeros(n_steps + 1)
        y[0] = y0
        for i in range(n_steps):
            y[i+1] = y[i] + h * f(t[i], y[i])
    else:
        y = np.zeros((n_steps + 1, len(y0)))
        y[0] = y0
        for i in range(n_steps):
            y[i+1] = y[i] + h * f(t[i], y[i])
    
    return t, y

# Define ODE functions
def f_a(t, y):
    return 0.5 * y

def f_b(t, y):
    return 2*t + 3*y

def f_c(t, x):
    x1, x2 = x
    return np.array([x2, -x1])

def f_d(t, x):
    x1, x2 = x
    return np.array([x2, 4*x1])

# Parameters
t_span = [0, 10]
h = 0.025

# Initial conditions
y0_a = 1
y0_b = -2
x0_c = np.array([1, 0])
x0_d = np.array([1, 1])


def plot_euler_solutions():
    # Compute solutions
    t_a, y_a = euler_system(f_a, y0_a, t_span, h)
    t_b, y_b = euler_system(f_b, y0_b, t_span, h)
    t_c, x_c = euler_system(f_c, x0_c, t_span, h)
    t_d, x_d = euler_system(f_d, x0_d, t_span, h)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Numerical Solutions by Euler Method (h=0.025)', fontsize=14)

    # Plot a
    axes[0,0].plot(t_a, y_a, 'b-', linewidth=1.5)
    axes[0,0].set_title("a) y' = 0.5y, y(0)=1")
    axes[0,0].set_xlabel('t')
    axes[0,0].set_ylabel('y(t)')
    axes[0,0].grid(True)

    # Plot b
    axes[0,1].plot(t_b, y_b, 'r-', linewidth=1.5)
    axes[0,1].set_title("b) y' = 2t + 3y, y(0)=-2")
    axes[0,1].set_xlabel('t')
    axes[0,1].set_ylabel('y(t)')
    axes[0,1].grid(True)

    # Plot c
    axes[1,0].plot(t_c, x_c[:, 0], 'g-', label='x1(t)', linewidth=1.5)
    axes[1,0].plot(t_c, x_c[:, 1], 'm-', label='x2(t)', linewidth=1.5)
    axes[1,0].set_title("c) x1'=x2, x2'=-x1, x1(0)=1, x2(0)=0")
    axes[1,0].set_xlabel('t')
    axes[1,0].set_ylabel('x(t)')
    axes[1,0].legend()
    axes[1,0].grid(True)

    # Plot d
    axes[1,1].plot(t_d, x_d[:, 0], 'c-', label='x1(t)', linewidth=1.5)
    axes[1,1].plot(t_d, x_d[:, 1], 'y-', label='x2(t)', linewidth=1.5)
    axes[1,1].set_title("d) x1'=x2, x2'=4x1, x1(0)=1, x2(0)=1")
    axes[1,1].set_xlabel('t')
    axes[1,1].set_ylabel('x(t)')
    axes[1,1].legend()
    axes[1,1].grid(True)

    plt.tight_layout()
    plt.show()

