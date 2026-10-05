"""
PW2 Lab B Part 5 (bonus) -- find a titration's equivalence point.

titration.csv holds a titration curve: pH versus the volume of base added.
The equivalence point is the volume where the pH changes fastest (the steep
jump). Numerically, that is where the SLOPE of the pH curve is largest.
Run:  python titration.py
"""
import numpy as np
import matplotlib.pyplot as plt

# TODO 1: read titration.csv (columns volume_base, pH) into arrays V, pH.
data = np.loadtxt("titration.csv", delimiter=",", skiprows=1)
V = data[:, 0]
pH = data[:, 1]

# TODO 2: compute the slope of the pH curve with np.gradient(pH, V), and find
#         the volume where that slope is largest (np.argmax). That is the
#         equivalence point. Print it.
slope = np.gradient(pH, V)
max_slope_idx = np.argmax(slope)
equivalence_volume = V[max_slope_idx]
print(f"Equivalence point volume: {equivalence_volume:.2f} mL")

# TODO 3: make two plots side by side: (left) pH vs volume with a line at the
#         equivalence point; (right) the slope vs volume, showing it peaks
#         at the equivalence point. Save as titration.png.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: pH curve
ax1.plot(V, pH, color="blue", label="Titration Curve")
ax1.axvline(equivalence_volume, color="red", linestyle="--", label=f"Equivalence Point ≈ {equivalence_volume:.1f} mL")
ax1.set_xlabel("Volume of Base Added (mL)")
ax1.set_ylabel("pH")
ax1.set_title("pH vs Volume")
ax1.legend()
ax1.grid(True)

# Right plot: Slope curve
ax2.plot(V, slope, color="green", label="d(pH)/d(V)")
ax2.axvline(equivalence_volume, color="red", linestyle="--", label=f"Peak Slope")
ax2.set_xlabel("Volume of Base Added (mL)")
ax2.set_ylabel("Slope (dpH/dV)")
ax2.set_title("Rate of pH Change")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("titration.png")
plt.show()