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



# Relationship Between Step Size ($\Delta t$) and Error

## 1. Does a larger $\Delta t$ mean more error?
**Yes.** As we found in Part (b), the local error is:
$$\text{Error} \approx C(\Delta t)^2$$

Because the error depends on the **square** of the step size, the penalty for a large $\Delta t$ is significant:
* If you **double** the step size ($2\Delta t$), the local error increases by **4 times** ($2^2$).
* If you **halve** the step size ($\frac{1}{2}\Delta t$), the local error drops to **1/4** of its previous value.



## 2. Is the solution to make $\Delta t$ "small enough"?
In theory, yes. By making $\Delta t$ very small, the Euler approximation stays much closer to the true tangent of the curve, minimizing the "drift." 

However, in computational engineering, there is a **trade-off** known as the "Computational Cost vs. Accuracy" balance:

| Factor | Small $\Delta t$ | Large $\Delta t$ |
| :--- | :--- | :--- |
| **Accuracy** | High (Low error) | Low (High error) |
| **Computation Speed** | Slow (More steps required) | Fast (Fewer steps) |
| **Stability** | Generally more stable | Can lead to "overshooting" or divergence |

## 3. Local vs. Global Error
It is important to distinguish between the error of one step and the error of the whole simulation:

1.  **Local Truncation Error (LTE):** The error in a single step, which we found is $O(\Delta t^2)$.
2.  **Global Truncation Error (GTE):** The total accumulated error after many steps.
    * To reach a final time $T$, you need $n = T / \Delta t$ steps.
    * Total Error $\approx n \times \text{LTE}$
    * Total Error $\approx (T / \Delta t) \times O(\Delta t^2) = O(\Delta t^1)$

This confirms that the Euler method is a **First-Order Method**. Even if you make $\Delta t$ small, the error only decreases linearly with the number of steps.

## 4. Practical Engineering Solution
Because Euler's method requires an extremely small $\Delta t$ to be accurate (which is computationally expensive), engineers usually move to **Higher-Order Methods** such as:
* **Runge-Kutta 2nd Order (RK2):** Global error is $O(\Delta t^2)$.
* **Runge-Kutta 4th Order (RK4):** Global error is $O(\Delta t^4)$. 

These methods allow you to use a much larger $\Delta t$ while maintaining a much higher level of precision than Euler.


# The Iterative Nature of the Euler Method

## 1. The "Step-by-Step" Logic
You are correct that $x_0$ is the only value we know for certain. Every subsequent value is an estimate based on a previous estimate. 

The algorithm looks like this:
1.  **Step 1:** Use $x_0$ to find $x_1$:
    $$x_1 = x_0 + f(x_0)\Delta t$$
2.  **Step 2:** Use $x_1$ (which is already an approximation) to find $x_2$:
    $$x_2 = x_1 + f(x_1)\Delta t$$
3.  **Step 3:** Use $x_2$ to find $x_3$:
    $$x_3 = x_2 + f(x_2)\Delta t$$

In general, the formula for any step $n$ is:
$$x_{n+1} = x_n + f(x_n)\Delta t$$

## 2. Why "Small $\Delta t$" is a Double-Edged Sword
Your observation about making $\Delta t$ "small enough" leads to a very famous problem in numerical analysis:

* **The Benefit:** A smaller $\Delta t$ makes each individual step more accurate (the "drift" from the tangent line is smaller).
* **The Risk:** A smaller $\Delta t$ means you need **more steps** to reach your target time $T$. 
    * If you decrease $\Delta t$ by 10x, you must perform 10x more calculations.
    * Since every step has a tiny bit of error, taking more steps gives the error more opportunities to "compound" or grow.



## 3. Visualizing the "Inherited" Error
Think of it like walking in a dark room where you can only see 1 foot in front of you:
* At $t_0$, you are standing at the **Exact** start ($x_0$).
* You take one step. You are now slightly off the path.
* For your **second** step, you calculate your direction (the slope) based on your **current (wrong) position**. 
* Because your position is wrong, your slope $f(x_n)$ is also slightly wrong. 

This is why the Euler method usually "drifts" away from the true solution curve over time.

---

## 4. Python Demonstration: Error Accumulation
The following code shows how the approximation gets worse as time goes on because it is building on previous errors.



# Local vs. Global Error: The "Step-by-Step" Reality

## 1. Local Truncation Error (LTE)
This is exactly what you just solved in the math problem. 
* **Definition:** The error made in a **single step**.
* **Assumption:** We assume that at the start of the step ($t_n$), our value $x_n$ is **perfectly correct**.
* **Formula:** As you derived, $\text{Local Error} \approx \frac{1}{2} f'(x_n)f(x_n) (\Delta t)^2$.
* **Big O Notation:** $O(\Delta t^2)$.

## 2. Global Truncation Error (GTE)
This is the error that actually matters to an engineer. It is the difference between the final estimated point and the true solution after many steps.
* **Definition:** The **accumulated** error over the entire interval $[t_0, T]$.
* **The "Inheritance" Problem:** Since $x_{n+1}$ is calculated using $x_n$, and $x_n$ is already wrong, the error usually grows exponentially or oscillates.
* **Big O Notation:** $O(\Delta t^1)$.



---

## 3. Why did the power of $\Delta t$ change from 2 to 1?
This is often the most confusing part for students. Here is the conceptual "Hand-Calculation":

1.  Suppose you want to simulate from $t=0$ to $t=1$.
2.  If your step size $\Delta t$ is **0.1**, you must take **10 steps**.
3.  Each step has an error of $(\Delta t)^2 = (0.1)^2 = 0.01$.
4.  Total Error $\approx (\text{Number of Steps}) \times (\text{Error per Step})$
5.  Total Error $\approx 10 \times 0.01 = \mathbf{0.1}$.

Notice that the final error ($0.1$) is the same magnitude as our step size ($\Delta t$). This is why Euler is called a **First-Order Method**.

---

## 4. Summary Table

| Term | Scope | Order | What it tells you |
| :--- | :--- | :--- | :--- |
| **Local Error** | One Step | $O(\Delta t^2)$ | How much you drift in one "leap". |
| **Global Error** | Full Simulation | $O(\Delta t^1)$ | How much you can trust your final result. |

---

## 5. Visualizing the "Drift" with Python
This script compares the **Local Error** (error in just the first step) versus the **Global Error** (total error at the end) as we change $\Delta t$.