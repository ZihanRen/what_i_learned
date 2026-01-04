
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

import numpy as np
import matplotlib.pyplot as plt

# ODE: dx/dt = x (The exact solution is x = e^t)
def f(x): return x

def euler_solve(x0, t_end, dt):
    t_values = np.arange(0, t_end + dt, dt)
    x_values = np.zeros(len(t_values))
    x_values[0] = x0
    
    for i in range(1, len(t_values)):
        # x_n = x_{n-1} + f(x_{n-1}) * dt
        x_values[i] = x_values[i-1] + f(x_values[i-1]) * dt
    return t_values, x_values

# Parameters
x0 = 1
t_end = 4
dt_large = 1.0
dt_small = 0.2

# Get solutions
t_exact = np.linspace(0, t_end, 100)
x_exact = np.exp(t_exact)

t_l, x_l = euler_solve(x0, t_end, dt_large)
t_s, x_s = euler_solve(x0, t_end, dt_small)

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(t_exact, x_exact, 'k', label='Exact Solution ($e^t$)', linewidth=2)
plt.step(t_l, x_l, 'r-o', where='post', label=f'Euler (Large dt={dt_large})', alpha=0.7)
plt.plot(t_s, x_s, 'b-s', label=f'Euler (Small dt={dt_small})', markersize=4)

plt.title("How Error Accumulates Step-by-Step")
plt.xlabel("Time (t)")
plt.ylabel("Value (x)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# %%

import numpy as np
import pandas as pd

# ODE: dx/dt = x, x(0) = 1. Exact: x(t) = e^t
def f(x): return x
def exact(t): return np.exp(t)

def get_errors(dt):
    # Local Error (First step only)
    x1_euler = 1 + f(1) * dt
    x1_exact = exact(dt)
    local_err = abs(x1_exact - x1_euler)
    
    # Global Error (Total error at t=1.0)
    steps = int(1.0 / dt)
    x_n = 1.0
    for _ in range(steps):
        x_n = x_n + f(x_n) * dt
    global_err = abs(exact(1.0) - x_n)
    
    return local_err, global_err

# Test different step sizes
dts = [0.1, 0.05, 0.025, 0.0125]
results = []

for dt in dts:
    loc, glob = get_errors(dt)
    results.append({"dt": dt, "Local Error": loc, "Global Error": glob})

df = pd.DataFrame(results)
print(df)

# Observe: 
# When dt is halved (0.1 -> 0.05):
# Local Error drops by ~4x (0.005 -> 0.0012) -> O(dt^2)
# Global Error drops by ~2x (0.12 -> 0.06) -> O(dt^1)
# %%
