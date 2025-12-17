import numpy as np
import matplotlib.pyplot as plt
from math import exp, sin, cos, log, sqrt, atan

def f1(x):
    return exp(-x/2)

def f2(x):
    return (sin(3*x**4/5))**3

def f3(x):
    return (cos(x/(x+1)))**2

def f4(x):
    return log(x + sqrt(4 + x**2))

def f5(x):
    return (x * atan(2*x)) / (x**2 + 4)

def derivative_numerical(f, a, b, h):
    x_vals = np.arange(a, b + h/2, h)
    y_vals = np.array([f(x) for x in x_vals])
    dydx = np.zeros_like(x_vals)
    dydx[:-1] = (y_vals[1:] - y_vals[:-1]) / h
    dydx[-1] = dydx[-2]
    return x_vals, dydx

def derivative_analytic1(x):
    return -0.5 * exp(-x/2)

def derivative_analytic2(x):
    inner = 3*x**4/5
    return 3*(sin(inner))**2 * cos(inner) * (12*x**3/5)

def derivative_analytic3(x):
    arg = x/(x+1)
    return -2*cos(arg)*sin(arg) * (1/(x+1)**2)

def derivative_analytic4(x):
    return 1/sqrt(4 + x**2)

def derivative_analytic5(x):
    num = atan(2*x)*(x**2+4) - x*(2*x/(4*x**2+1))
    den = (x**2+4)**2
    return num/den

functions = [f1, f2, f3, f4, f5]
derivatives = [derivative_analytic1, derivative_analytic2, derivative_analytic3, derivative_analytic4, derivative_analytic5]
intervals = [(0, 1), (2, 15), (-5, 5)]
steps = [0.01, 0.005]
func_names = ['exp(-x/2)', 'sin^3(3x^4/5)', 'cos^2(x/(x+1))', 'ln(x+sqrt(4+x^2))', '(x*arctg2x)/(x^2+4)']


def plot_derivative_comparison():
    fig, axes = plt.subplots(len(functions), len(intervals), figsize=(18, 22), constrained_layout=True)

    for i, (func, deriv, name) in enumerate(zip(functions, derivatives, func_names)):
        for j, (a, b) in enumerate(intervals):
            ax = axes[i, j]
            x_fine = np.linspace(a, b, 1000)
            y_analytic = [deriv(x) for x in x_fine]
            ax.plot(x_fine, y_analytic, 'k-', linewidth=2, label='Analytic')
            
            for h in steps:
                x_num, y_num = derivative_numerical(func, a, b, h)
                ax.plot(x_num, y_num, '--', linewidth=1.5, label=f'h={h}')
            
            ax.set_xlabel('x')
            ax.set_ylabel("y'")
            ax.set_title(f'{name}\nInterval [{a}, {b}]')
            ax.grid(True, alpha=0.3)
            if j == len(intervals)-1:
                ax.legend(loc='best')

    plt.suptitle('Numerical vs Analytic Derivatives', fontsize=16, y=1.02)
    plt.show()