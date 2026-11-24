import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Look at the record (in mm/s)
# ==========================================
d = pd.read_csv("data/seismogram_A.csv")
t = d["time_s"].to_numpy()
# Convert ground velocity from m/s to mm/s (* 1000)
z_mm = d["Z"].to_numpy() * 1000.0
n_mm = d["N"].to_numpy() * 1000.0
e_mm = d["E"].to_numpy() * 1000.0

fs = 1.0 / (t[1] - t[0])
print(f"Sampling rate fs = {fs:.2f} Hz, total samples: {len(t)}")

# 1.1 All three channels stacked with shared time axis
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 7))
channels = [("Z (vertical)", z_mm), ("N (north-south)", n_mm), ("E (east-west)", e_mm)]

for ax, (label, sig) in zip(axes, channels):
    ax.plot(t, sig, lw=0.7)
    ax.set_ylabel(f"{label}\n(mm/s)")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (s)")
axes[0].set_title("Seismogram A — Three Channels")
plt.tight_layout()
plt.savefig("step1_channels_full.png", dpi=150)
plt.show()

# 1.2 Zoom in on the first 60 seconds (pre-event background noise)
mask_60 = t <= 60.0
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 7))

for ax, (label, sig) in zip(axes, channels):
    ax.plot(t[mask_60], sig[mask_60], lw=0.7)
    ax.set_ylabel(f"{label}\n(mm/s)")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (s)")
axes[0].set_title("First 60 s — Background Noise Check")
plt.tight_layout()
plt.savefig("step1_first_60s.png", dpi=150)
plt.show()

# Step 2: Spectrum of Z (FFT)
# ==========================================
N = len(z_mm)
X = np.fft.fft(z_mm)
freqs = np.fft.fftfreq(N, d=1.0 / fs)
amp = np.abs(X) / N * 2.0

# 2(a) Amplitude vs array index, then vs fftfreq (Hz)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(np.arange(N), amp, lw=0.7)
ax1.set_title("Z Spectrum vs. Array Index")
ax1.set_xlabel("Index k")
ax1.set_ylabel("Amplitude (mm/s)")
ax1.grid(True, alpha=0.3)

ax2.plot(freqs, amp, lw=0.7)
ax2.set_title("Z Spectrum vs. Frequency (fftfreq)")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Amplitude (mm/s)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("step2a_fft_index_vs_freq.png", dpi=150)
plt.show()

# 2(b) Log-log spectrum: first 60 s (quiet) vs full record (positive frequencies)
pos = freqs > 0

z_quiet = z_mm[mask_60]
N_q = len(z_quiet)
X_q = np.fft.fft(z_quiet)
freqs_q = np.fft.fftfreq(N_q, d=1.0 / fs)
amp_q = np.abs(X_q) / N_q * 2.0
pos_q = freqs_q > 0

plt.figure(figsize=(9, 5))
plt.loglog(freqs_q[pos_q], amp_q[pos_q], label="First 60 s (background noise)", alpha=0.8)
plt.loglog(freqs[pos], amp[pos], label="Full record (including earthquake)", alpha=0.8)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (mm/s)")
plt.title("Z Channel Spectrum (log–log, f > 0)")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("step2b_loglog_spectrum.png", dpi=150)
plt.show()