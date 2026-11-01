import matplotlib.pyplot as plt
import numpy as np

# 1. Load data from trajectory.csv
data = np.loadtxt('trajectory.csv', delimiter=',', skiprows=1)
t = data[:, 0]
x = data[:, 1]
y = data[:, 2]

# 2. Compute velocity components in x and y using np.gradient
v_x = np.gradient(x, t)
v_y = np.gradient(y, t)

# 3. Compute speed using the formula sqrt(v_x^2 + v_y^2)
speed = np.sqrt(v_x**2 + v_y**2)

# 4. Create a figure with two panels
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# Panel 1: Motion trajectory (x vs y)
axes[0].plot(x, y, color='purple', label='Trajectory (x vs y)')
axes[0].set_xlabel('X Position (m)')
axes[0].set_ylabel('Y Position (m)')
axes[0].set_title('2D Tracked Trajectory')
axes[0].grid(True)
axes[0].legend()

# Panel 2: Speed over time
axes[1].plot(t, speed, color='teal', label='Speed v(t)')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Speed (m/s)')
axes[1].set_title('Speed vs Time')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.savefig('trajectory_plot.png')
print(
    'Bonus analysis completed successfully! Figure saved as trajectory_plot.png'
)