# Solution to Problem 2.8.7: Error Estimate for Euler Method

## Problem Statement
We are analyzing the error in the Euler method for the differential equation:
$$\dot{x} = f(x)$$
with the initial condition $x(t_0) = x_0$.

We want to compare:
1.  **The Exact Solution:** $x(t_1)$ where $t_1 = t_0 + \Delta t$.
2.  **The Euler Approximation:** $x_{Euler} = x_0 + f(x_0)\Delta t$.

---

## Part (a): Taylor Series Expansion

**Goal:** Expand the exact solution $x(t_1) = x(t_0 + \Delta t)$ as a Taylor series in $\Delta t$ through terms of order $O(\Delta t^2)$.

### 1. General Taylor Series Formula
The Taylor series expansion for a function $x(t)$ around a point $t_0$ is given by:
$$x(t_0 + \Delta t) = x(t_0) + \Delta t \cdot \dot{x}(t_0) + \frac{(\Delta t)^2}{2!} \cdot \ddot{x}(t_0) + O(\Delta t^3)$$

### 2. Determine the First Derivative Term $\dot{x}(t_0)$
From the definition of our ODE, the rate of change of $x$ is given by the function $f(x)$:
$$\dot{x}(t) = f(x(t))$$

Evaluating this at $t = t_0$:
$$\dot{x}(t_0) = f(x_0)$$

### 3. Determine the Second Derivative Term $\ddot{x}(t_0)$
To find the second derivative $\ddot{x}$, we differentiate the ODE equation $\dot{x} = f(x)$ with respect to time $t$.
$$\ddot{x} = \frac{d}{dt} [ f(x) ]$$

Since $f$ depends on $x$, and $x$ depends on $t$, we must use the **Chain Rule**:
$$\frac{d}{dt} f(x) = \frac{df}{dx} \cdot \frac{dx}{dt}$$
$$\frac{d}{dt} f(x) = f'(x) \cdot \dot{x}$$

Substitute $\dot{x} = f(x)$ back into the equation:
$$\ddot{x} = f'(x) \cdot f(x)$$

Evaluating this at $t = t_0$:
$$\ddot{x}(t_0) = f'(x_0)f(x_0)$$

### 4. Final Expansion Result
Substitute the expressions from Steps 2 and 3 back into the general Taylor Series formula:
$$x(t_1) = x_0 + \Delta t [f(x_0)] + \frac{(\Delta t)^2}{2} [f'(x_0)f(x_0)] + O(\Delta t^3)$$

**Answer for (a):**
$$x(t_1) = x_0 + f(x_0)\Delta t + \frac{1}{2}f'(x_0)f(x_0)(\Delta t)^2 + O(\Delta t^3)$$

---

## Part (b): Local Error Estimate

**Goal:** Show that the local error $|x(t_1) - x_{Euler}| \approx C(\Delta t)^2$ and find the constant $C$.

### 1. Define the Local Error
The local truncation error is the difference between the exact Taylor expansion (derived in Part a) and the Euler approximation step.
$$\text{Error} = | x_{\text{exact}}(t_1) - x_{\text{Euler}} |$$

### 2. Compare the Equations
**The Exact Solution (from Part a):**
$$x_{\text{exact}} = x_0 + f(x_0)\Delta t + \frac{1}{2}f'(x_0)f(x_0)(\Delta t)^2 + O(\Delta t^3)$$

**The Euler Approximation (Definition):**
$$x_{\text{Euler}} = x_0 + f(x_0)\Delta t$$

### 3. Subtract to Find the Difference
$$x_{\text{exact}} - x_{\text{Euler}} = \left[ x_0 + f(x_0)\Delta t + \frac{1}{2}f'(x_0)f(x_0)(\Delta t)^2 \right] - \left[ x_0 + f(x_0)\Delta t \right]$$

The terms $x_0$ and $f(x_0)\Delta t$ appear in both expressions and cancel each other out.
$$x_{\text{exact}} - x_{\text{Euler}} = \frac{1}{2}f'(x_0)f(x_0)(\Delta t)^2 + O(\Delta t^3)$$

### 4. Identify the Constant C
The error is dominated by the $(\Delta t)^2$ term. We can write this as $C(\Delta t)^2$.
Comparing the coefficients, we find:

**Answer for (b):**
The explicit expression for the constant $C$ is:
$$C = \frac{1}{2} f'(x_0) f(x_0)$$