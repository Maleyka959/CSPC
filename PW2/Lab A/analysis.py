"""
PW2 Lab A -- Motion from tracking data.

Read noisy free-fall position measurements, then:
- differentiate once  -> velocity
- differentiate twice -> acceleration (should be ~ constant -g, but noisy!)
- integrate the acceleration back up -> recover velocity and position

Complete the TODOs. Run:  python analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# TODO 1: read freefall.csv into arrays t and y
#         (hint: np.loadtxt with a comma delimiter, skipping the header)
data = np.loadtxt('freefall.csv', delimiter=',', skiprows=1)
t = data[:, 0]
y = data[:, 1]

# TODO 2: compute velocity v = derivative of y w.r.t. t   (np.gradient)
#         and acceleration a = derivative of v w.r.t. t    (np.gradient again)
#         Print the mean acceleration. Is it close to -9.81? Is it noisy?
v = np.gradient(y, t)
a = np.gradient(v, t)

mean_a = np.mean(a)
std_a = a.std()
print(f"Mean acceleration: {mean_a:.2f} m/s^2")
print(f"Standard deviation: {std_a:.2f} m/s^2")

# TODO 3: integrate a back up to recover velocity and position
#         (hint: cumulative_trapezoid(a, t, initial=0) + v[0], then again)
# Integrate acceleration to get recovered velocity (v_rec)
v_rec = cumulative_trapezoid(a, t, initial=0) + v[0]

# Integrate recovered velocity to get recovered position (y_rec)
y_rec = cumulative_trapezoid(v_rec, t, initial=0) + y[0]

# Calculate the largest difference between original and recovered position
max_diff = np.max(np.abs(y - y_rec))
print(f"Largest position difference: {max_diff:.2f} meters")

# TODO 4: make a figure with 3 stacked panels: position, velocity, acceleration
#         vs time. Mark the true -9.81 line on the acceleration panel.
#         Save it as motion.png
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Panel 1: Position
axes[0].plot(t, y, label='Measured y', color='black', alpha=0.6)
axes[0].plot(t, y_rec, label='Recovered y (from integrated a)', color='blue', linestyle='--')
axes[0].set_ylabel('Position (m)')
axes[0].legend()
axes[0].grid(True)

# Panel 2: Velocity
axes[1].plot(t, v, label='Velocity (from gradient y)', color='orange')
axes[1].plot(t, v_rec, label='Recovered v (from integrated a)', color='green', linestyle='--')
axes[1].set_ylabel('Velocity (m/s)')
axes[1].legend()
axes[1].grid(True)

# Panel 3: Acceleration
axes[2].plot(t, a, label='Acceleration (noisy)', color='red', alpha=0.7)
axes[2].axhline(-9.81, color='black', linestyle=':', label='True -9.81 m/s^2')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Acceleration (m/s^2)')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('motion.png')
print("Figure saved as motion.png successfully!")