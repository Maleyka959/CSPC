import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Load seismogram A
d = pd.read_csv("data/seismogram_A.csv")
t = d["time_s"].to_numpy()
z_raw = d["Z"].to_numpy()  # m/s
n_raw = d["N"].to_numpy()  # m/s
e_raw = d["E"].to_numpy()  # m/s
fs = 1.0 / (t[1] - t[0])

# Step 7: Spectrogram of Raw Z Channel (up to 10 Hz)
# ==========================================
f, ts, S = spectrogram(z_raw, fs=fs, nperseg=256)
freq_mask = f <= 10.0

plt.figure(figsize=(11, 5))
mesh = plt.pcolormesh(ts, f[freq_mask], 10.0 * np.log10(S[freq_mask, :] + 1e-30), shading="auto")
plt.colorbar(mesh, label="Power Spectral Density (dB)")
plt.ylim(0.0, 10.0)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram of Raw Z Channel (0–10 Hz)")
plt.tight_layout()
plt.savefig("step7_spectrogram_z.png", dpi=150)
plt.show()

# Step 8: PGV and Magnitude M
# ==========================================
# PGV is max of sqrt(N^2 + E^2) over the whole raw record, in cm/s (1 m/s = 100 cm/s)
horiz_raw_cms = np.sqrt(n_raw**2 + e_raw**2) * 100.0
pgv_cms = float(np.max(horiz_raw_cms))

# Distance R from Step 6 (306.13 km)
R_km = 306.13

# Magnitude formula: M = (log10(PGV [cm/s]) + log10(R [km]) + 1.0) / 0.55
M = (np.log10(pgv_cms) + np.log10(R_km) + 1.0) / 0.55

# Factor increase in PGV at the same distance for +1.0 magnitude unit:
# Delta M = 1 => Delta(log10(PGV)) = 0.55 => PGV multiplies by 10^0.55
pgv_factor_plus1M = 10.0 ** 0.55

print(f"PGV (raw horizontal max): {pgv_cms:.6f} cm/s")
print(f"Estimated Magnitude M (using R = {R_km:.2f} km): {M:.2f}")
print(f"PGV multiplier for +1.0 magnitude unit at same distance: {pgv_factor_plus1M:.2f}x (10^0.55)")