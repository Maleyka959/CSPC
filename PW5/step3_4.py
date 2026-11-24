import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load seismogram A and convert velocity from m/s to mm/s
d = pd.read_csv("data/seismogram_A.csv")
t = d["time_s"].to_numpy()
z = d["Z"].to_numpy() * 1000.0
n = d["N"].to_numpy() * 1000.0
e = d["E"].to_numpy() * 1000.0

fs = 1.0 / (t[1] - t[0])

# Step 3: Inverse FFT & Manual Band-Pass
# ==========================================
# 3(a) Verify round-trip FFT -> IFFT
z_roundtrip = np.fft.ifft(np.fft.fft(z)).real
is_close = np.allclose(z, z_roundtrip)
print(f"Step 3(a) np.allclose(z, ifft(fft(z)).real): {is_close}")

# 3(b) Band-pass filter function
def bandpass(x, fs, a, b):
    N = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(N, d=1.0 / fs)
    X[(np.abs(f) < a) | (np.abs(f) > b)] = 0.0
    return np.fft.ifft(X).real

a, b = 0.5, 10.0
z_clean = bandpass(z, fs, a, b)
n_clean = bandpass(n, fs, a, b)
e_clean = bandpass(e, fs, a, b)

# 3(c) Plot Z - clean ("what was removed") for the whole record
z_removed = z - z_clean

plt.figure(figsize=(11, 4.5))
plt.plot(t, z_removed, lw=0.7)
plt.title(f"Removed Signal (Z - clean) Outside {a}–{b} Hz")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (mm/s)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("step3c_removed_z.png", dpi=150)
plt.show()

# Step 4: Envelopes & Window Length Comparison
# ==========================================
def envelope(x, fs, win_sec):
    w = max(1, int(round(win_sec * fs)))
    kernel = np.ones(w) / w
    return np.convolve(np.abs(x), kernel, mode="same")

# Envelopes at 0.5 s for filtered Z and filtered total horizontal motion
env_z_05 = envelope(z_clean, fs, 0.5)
horiz_clean = np.sqrt(n_clean**2 + e_clean**2)
env_h_05 = envelope(horiz_clean, fs, 0.5)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11, 6.5))
ax1.plot(t, np.abs(z_clean), lw=0.5, alpha=0.5, label="|Z_clean|")
ax1.plot(t, env_z_05, lw=1.2, label="Envelope Z (0.5 s)")
ax1.set_ylabel("Z (mm/s)")
ax1.set_title("Filtered Envelopes (Window = 0.5 s)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(t, horiz_clean, lw=0.5, alpha=0.5, label="sqrt(N^2 + E^2)")
ax2.plot(t, env_h_05, lw=1.2, label="Envelope Horizontal (0.5 s)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Horizontal (mm/s)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("step4_envelopes_0p5s.png", dpi=150)
plt.show()

# Compare window sizes 0.05 s, 0.5 s, and 5 s on filtered Z
env_z_005 = envelope(z_clean, fs, 0.05)
env_z_50 = envelope(z_clean, fs, 5.0)

plt.figure(figsize=(11, 5))
plt.plot(t, env_z_005, lw=0.8, alpha=0.8, label="Window = 0.05 s (too short)")
plt.plot(t, env_z_05, lw=1.2, label="Window = 0.5 s")
plt.plot(t, env_z_50, lw=1.5, label="Window = 5.0 s (too long)")
plt.title("Effect of Smoothing Window on Filtered Z Envelope")
plt.xlabel("Time (s)")
plt.ylabel("Envelope Amplitude (mm/s)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("step4_window_comparison.png", dpi=150)
plt.show()