import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load seismogram A and convert velocity from m/s to mm/s
d = pd.read_csv("data/seismogram_A.csv")
t = d["time_s"].to_numpy()
z = d["Z"].to_numpy() * 1000.0
n = d["N"].to_numpy() * 1000.0
e = d["E"].to_numpy() * 1000.0
fs = 1.0 / (t[1] - t[0])

# Band-pass filter (0.5 - 10 Hz) from Step 3
def bandpass(x, fs, a, b):
    N = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(N, d=1.0 / fs)
    X[(np.abs(f) < a) | (np.abs(f) > b)] = 0.0
    return np.fft.ifft(X).real

z_clean = bandpass(z, fs, 0.5, 10.0)
n_clean = bandpass(n, fs, 0.5, 10.0)
e_clean = bandpass(e, fs, 0.5, 10.0)
horiz_clean = np.sqrt(n_clean**2 + e_clean**2)

# Envelope with a 0.5 s window from Step 4
def envelope(x, fs, win_sec):
    w = max(1, int(round(win_sec * fs)))
    kernel = np.ones(w) / w
    return np.convolve(np.abs(x), kernel, mode="same")

env_z = envelope(z_clean, fs, 0.5)
env_h = envelope(horiz_clean, fs, 0.5)

# Step 5: P- and S-Wave Arrival Times
# ==========================================
# P-wave arrival: first time filtered Z envelope exceeds 5% of its maximum
p_level = 0.05 * np.max(env_z)
p_idx = int(np.argmax(env_z > p_level))
t_P = t[p_idx]

# S-wave arrival: first time after t_P + 2 s that horizontal envelope exceeds 10% of its maximum
s_level = 0.10 * np.max(env_h)
after_mask = t >= (t_P + 2.0)
after_indices = np.where(after_mask)[0]
s_rel_idx = int(np.argmax(env_h[after_mask] > s_level))
s_idx = int(after_indices[s_rel_idx])
t_S = t[s_idx]

# Time of horizontal envelope maximum using find_peaks
peaks, _ = find_peaks(env_h)
max_peak_idx = peaks[np.argmax(env_h[peaks])] if len(peaks) > 0 else int(np.argmax(env_h))
t_max_h = t[max_peak_idx]

print(f"t_P (P-wave arrival): {t_P:.2f} s")
print(f"t_S (S-wave arrival): {t_S:.2f} s")
print(f"t_S - t_P: {t_S - t_P:.2f} s")
print(f"Horizontal envelope peak time: {t_max_h:.2f} s (difference from t_S: {t_max_h - t_S:.2f} s)")

# Plot all three channels with t_P and t_S marked
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 7))
for ax, (label, sig) in zip(axes, [("Z (mm/s)", z), ("N (mm/s)", n), ("E (mm/s)", e)]):
    ax.plot(t, sig, lw=0.7)
    ax.axvline(t_P, color="tab:red", linestyle="--", lw=1.2, label=f"t_P = {t_P:.2f} s")
    ax.axvline(t_S, color="tab:green", linestyle="--", lw=1.2, label=f"t_S = {t_S:.2f} s")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Time (s)")
axes[0].set_title("P and S Wave Arrivals Across Channels")
plt.tight_layout()
plt.savefig("step5_arrivals.png", dpi=150)
plt.show()

# Step 6: Earthquake Distance and Origin Time
# ==========================================
v_P = 6.2  # km/s
v_S = 3.6  # km/s

factor = (1.0 / v_S) - (1.0 / v_P)
R_km = (t_S - t_P) / factor

# Change in R for a 1 s uncertainty in t_S
delta_R_km = 1.0 / factor

# Origin time
t_origin = t_P - (R_km / v_P)

print(f"\nEstimated distance R: {R_km:.2f} km")
print(f"Change in R for a 1 s uncertainty in t_S: +/- {delta_R_km:.2f} km")
print(f"Origin time: {t_origin:.2f} s")