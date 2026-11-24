from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def find_raw_data():
    for candidate in [Path("Raw Data.csv"), Path("data/Raw Data.csv")]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find 'Raw Data.csv' in current folder or data/.")


def bandpass(x, fs, a=0.5, b=15.0):
    n_samples = len(x)
    spectrum = np.fft.fft(x)
    freqs = np.fft.fftfreq(n_samples, d=1.0 / fs)
    # Using a > 0 removes the 0 Hz constant gravity component (~9.81 m/s^2)
    spectrum[(np.abs(freqs) < a) | (np.abs(freqs) > b)] = 0.0
    return np.fft.ifft(spectrum).real


def envelope(x, fs, win_sec=0.25):
    w = max(1, int(round(win_sec * fs)))
    kernel = np.ones(w) / w
    return np.convolve(np.abs(x), kernel, mode="same")


def main():
    csv_path = find_raw_data()
    df = pd.read_csv(csv_path)

    # Phyphox CSV uses "Time (s)" and "Acceleration z (m/s^2)"
    t_raw = df["Time (s)"].to_numpy()
    z_raw = df["Acceleration z (m/s^2)"].to_numpy()

    # Check uneven spacing in raw timestamps
    dt = np.diff(t_raw)
    print(f"Raw dt min: {np.min(dt):.6f} s, max: {np.max(dt):.6f} s, mean: {np.mean(dt):.6f} s")

    # Interpolate onto an even time grid
    fs = 1.0 / float(np.mean(dt))
    t_even = np.arange(t_raw[0], t_raw[-1], 1.0 / fs)
    z_even = np.interp(t_even, t_raw, z_raw)

    # Spectrum before filtering (shows 0 Hz gravity peak around 9.81 m/s^2)
    freqs = np.fft.rfftfreq(len(z_even), d=1.0 / fs)
    amp_spec = np.abs(np.fft.rfft(z_even)) / len(z_even)
    print(f"DC (0 Hz) amplitude in raw spectrum: {amp_spec[0]:.3f} m/s^2")

    # Filter out 0 Hz gravity and high-frequency noise
    z_clean = bandpass(z_even, fs, a=0.5, b=15.0)
    env = envelope(z_clean, fs, win_sec=0.25)

    # Detect the 4 drop peaks (heights in cm: 10, 20, 40, 80)
    heights_cm = np.array([10.0, 20.0, 40.0, 80.0])
    min_distance_samples = max(1, int(round(2.0 * fs)))
    peaks, _ = find_peaks(env, distance=min_distance_samples, prominence=0.05 * np.max(env))

    if len(peaks) >= 4:
        top4 = peaks[np.argsort(env[peaks])[-4:]]
        drop_peaks = np.sort(top4)
    else:
        drop_peaks = peaks

    print(f"Detected {len(drop_peaks)} drop peaks at times (s): {np.round(t_even[drop_peaks], 2)}")

    # Compute M_phone = log10(A / A1) and log10(m * g * h)
    m_kg = 1.5
    g = 9.81
    amps = env[drop_peaks]
    A1 = float(np.min(amps))
    m_phone = np.log10(amps / A1)
    energies_j = m_kg * g * (heights_cm / 100.0)
    log10_energy = np.log10(energies_j)

    for h_cm, t_pk, amp_val, m_ph, e_j, log_e in zip(
        heights_cm, t_even[drop_peaks], amps, m_phone, energies_j, log10_energy
    ):
        print(
            f"Drop {h_cm:4.0f} cm (t={t_pk:5.2f} s): A={amp_val:.4f} m/s^2, "
            f"M_phone={m_ph:.3f}, E={e_j:.2f} J, log10(mgh)={log_e:.3f}"
        )

    # Plot phone envelope with detected drops
    plt.figure(figsize=(11, 4.5))
    plt.plot(t_even, env, lw=1.0, label="Filtered Envelope (0.5-15 Hz)")
    plt.plot(t_even[drop_peaks], env[drop_peaks], "ro", label="Detected Drops")
    plt.xlabel("Time (s)")
    plt.ylabel("Envelope Amplitude (m/s^2)")
    plt.title("Phyphox Accelerometer Envelope and Detected Bottle Drops")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("bonus_phone_envelope.png", dpi=150)
    plt.show()

    # Plot M_phone vs log10(mgh)
    plt.figure(figsize=(7, 5))
    plt.plot(log10_energy, m_phone, "o-", lw=1.5)
    for h_cm, x_val, y_val in zip(heights_cm, log10_energy, m_phone):
        plt.annotate(f"{int(h_cm)} cm", (x_val, y_val), textcoords="offset points", xytext=(6, 4))
    plt.xlabel("log10(Drop Energy mgh) [log10(J)]")
    plt.ylabel("M_phone = log10(A / A1)")
    plt.title("Phone Magnitude vs. Log10 Drop Energy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("bonus_phone_magnitude_vs_energy.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
