"""
PW2 Lab B Part 2 -- three routes to a minimum.

Compare gradient descent, Newton, and SLSQP on two functions:
  2A: f(x) = (x-3)**2 + 1          (easy, one minimum at x=3)
  2B: g(x) = x**4 - 3*x**2 + x + 5 (harder, several stationary points)
Run:  python warmup.py
"""
import numpy as np
from scipy.optimize import newton, minimize

# ---------- 2A: easy convex function ----------
def f(x):   return (x-3)**2 + 1
def df(x):  return 2*(x-3)
def d2f(x): return 2.0

print("=== Part 2A: Easy Convex Function ===")

# (1) Gradient descent by hand
x = 0.0
lr = 0.1
for _ in range(100):
    step = lr * df(x)
    if abs(step) < 1e-6:
        break
    x = x - step
print(f"1) Gradient Descent: x = {x}")

# (2) scipy.optimize.newton
res_newton = newton(df, x0=0.0, fprime=d2f)
print(f"2) Newton: x = {res_newton}")

# (3) scipy.optimize.minimize with SLSQP
res_slsqp = minimize(f, x0=0.0, method="SLSQP")
print(f"3) SLSQP: x = {res_slsqp.x[0]}")


# ---------- 2B: harder landscape ----------
def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6

print("\n=== Part 2B: Harder Landscape ===")

for x0 in [0.0, 2.0]:
    print(f"\n--- Starting from x0 = {x0} ---")
    
    # (1) Gradient descent by hand
    x = x0
    lr = 0.01
    for _ in range(2000):
        step = lr * dg(x)
        if abs(step) < 1e-8:
            break
        x = x - step
    print(f"Gradient Descent -> x = {x:.5f}, g(x) = {g(x):.5f}")
    
    # (2) Newton's method
    try:
        res_n = newton(dg, x0=x0, fprime=d2g)
        d2_val = d2g(res_n)
        status = "minimum" if d2_val > 0 else "maximum/stationary"
        print(f"Newton -> x = {res_n:.5f}, g''(x) = {d2_val:.2f} ({status})")
    except Exception as e:
        print(f"Newton failed to converge from x0 = {x0}: {e}")
    
    # (3) SLSQP
    res_s = minimize(g, x0=x0, method="SLSQP")
    print(f"SLSQP -> x = {res_s.x[0]:.5f}, g(x) = {res_s.fun:.5f}")