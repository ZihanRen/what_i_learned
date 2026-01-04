
#%%
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Problem
# ODE: dx/dt = f(x) = x
# Initial condition: x0 = 1 at t0 = 0
def f(x):
    return x

def exact_sol(t):
    return np.exp(t)

# Parameters
t0 = 0
x0 = 1
dt = 1.0  # Large step size to make the error obvious visually

# 2. Calculate Values
# Exact value at t1
t1 = t0 + dt
x_exact = exact_sol(t1)

# Euler approximation
slope = f(x0)
x_euler = x0 + slope * dt

# Taylor Series (2nd Order) Approximation for visualization
# x_taylor ≈ x0 + dt*f(x0) + (dt^2/2)*f'(x0)f(x0)
# Here f(x)=x, so f'(x)=1. Thus f'(x0)f(x0) = 1 * 1 = 1
x_taylor = x0 + slope * dt + (dt**2 / 2) * 1 * 1

# 3. Plotting
t_range = np.linspace(t0, t1 + 0.2, 100)
plt.figure(figsize=(10, 6))

# Plot Exact Solution Curve
plt.plot(t_range, exact_sol(t_range), 'k-', linewidth=2, label=r'Exact Solution $x(t) = e^t$')

# Plot Euler Step (Tangent Line)
# The line equation is y = x0 + slope * (t - t0)
euler_line = x0 + slope * (t_range - t0)
plt.plot(t_range, euler_line, 'b--', linewidth=2, label=r'Euler Step (Linear Projection)')

# Plot Points
plt.scatter([t0], [x0], color='black', zorder=5, label='Start Point $(t_0, x_0)$')
plt.scatter([t1], [x_exact], color='green', zorder=5, label=f'Exact Value at $t_1$: {x_exact:.4f}')
plt.scatter([t1], [x_euler], color='red', zorder=5, label=f'Euler Approx at $t_1$: {x_euler:.4f}')

# Visualize the Error
plt.vlines(t1, x_euler, x_exact, color='red', linestyle=':', linewidth=2)
plt.text(t1 + 0.02, (x_exact + x_euler)/2, r'Local Error $\approx C(\Delta t)^2$', color='red', fontsize=12)

# Annotations
plt.title(f'Visualizing Euler Method Error (Step Size $\Delta t = {dt}$)')
plt.xlabel('Time $t$')
plt.ylabel('State $x$')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')

plt.show()
# %%
