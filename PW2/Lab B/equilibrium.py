"""
PW2 Lab B Part 4 -- chemical equilibrium via the equilibrium constant K.

Reaction  H2 + I2 <=> 2 HI, starting from 1 mol H2 and 1 mol I2.
As the reaction proceeds by an extent x:  H2 = 1-x,  I2 = 1-x,  HI = 2x.
At equilibrium the composition satisfies the equilibrium constant
        K = [HI]^2 / ([H2][I2]) = (2x)^2 / ((1-x)(1-x)).
Given K, find the extent x. Solve it TWO ways and compare.
Run:  python equilibrium.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 15.6
a = b = 1.0

# TODO 1: write k_imbalance(x) = (2x)^2/((a-x)(b-x)) - K.
#         It equals zero exactly at equilibrium.
def k_imbalance(x):
    return (2 * x)**2 / ((a - x) * (b - x)) - K

# TODO 2 (method 1): use scipy.optimize.newton to find the root of k_imbalance
#         (start x0=0.5). This is root-finding.
x_newton = newton(k_imbalance, x0=0.5)
print(f"Newton method equilibrium x: {x_newton:.4f}")

# TODO 3 (method 2): use scipy.optimize.minimize to minimise k_imbalance(x)**2
#         (method "SLSQP", bounds [(0, 0.999)], x0=[0.5]). Print both answers
#         and confirm they agree.
def objective(x):
    # x is passed as an array [x_val] by minimize
    val = x[0] if hasattr(x, '__len__') else x
    return k_imbalance(val)**2

result_slsqp = minimize(objective, x0=[0.5], method="SLSQP", bounds=[(0, 0.999)])
x_slsqp = result_slsqp.x[0]
print(f"SLSQP method equilibrium x: {x_slsqp:.4f}")

# Confirm they agree
print(f"Do they agree? Extent difference: {abs(x_newton - x_slsqp):.2e}")

# TODO 4: report the equilibrium amounts (H2, I2, HI), and plot how the three
#         amounts change with the extent x, marking the equilibrium. Save
#         equilibrium.png.
eq_h2 = a - x_newton
eq_i2 = b - x_newton
eq_hi = 2 * x_newton

print(f"\nEquilibrium amounts:")
print(f"  H2 = {eq_h2:.4f} mol")
print(f"  I2 = {eq_i2:.4f} mol")
print(f"  HI = {eq_hi:.4f} mol")

# Plotting
x_vals = np.linspace(0, 0.999, 200)
h2_vals = a - x_vals
i2_vals = b - x_vals
hi_vals = 2 * x_vals

plt.figure(figsize=(8, 5))
plt.plot(x_vals, h2_vals, label="H2", color="blue")
plt.plot(x_vals, i2_vals, label="I2", color="green")
plt.plot(x_vals, hi_vals, label="HI", color="red")
plt.axvline(x_newton, color="black", linestyle="--", label=f"Equilibrium x ≈ {x_newton:.3f}")
plt.xlabel("Extent of reaction (x)")
plt.ylabel("Amount (mol)")
plt.title("Chemical Equilibrium vs Reaction Extent")
plt.legend()
plt.grid(True)
plt.savefig("equilibrium.png")
plt.show()