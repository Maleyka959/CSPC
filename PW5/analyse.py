from pathlib import Path
import numpy as np
import pandas as pd


def bandpass(x, fs, a=0.5, b=12.0):
    n_samples = len(x)
    spectrum = np.fft.fft(x)
    freqs = np.fft.fftfreq(n_samples, d=1.0 / fs)
    spectrum[(np.abs(freqs) < a) | (np.abs(freqs) > b)] = 0.0
    return np.fft.ifft(spectrum).real


def envelope(x, fs, win_sec=0.5):
    w = max(1, int(round(win_sec * fs)))
    kernel = np.ones(w) / w
    return np.convolve(np.abs(x), kernel, mode="same")


def analyse(csv_file):
    d = pd.read_csv(csv_file)
    t = d["time_s"].to_numpy()
    z_raw = d["Z"].to_numpy()
    n_raw = d["N"].to_numpy()
    e_raw = d["E"].to_numpy()
    fs = 1.0 / (t[1] - t[0])

    # Velocity in mm/s for filtering and envelope picking
    z_mms = z_raw * 1000.0
    n_mms = n_raw * 1000.0
    e_mms = e_raw * 1000.0

    # Band-pass filter (0.5 - 10 Hz)
    z_clean = bandpass(z_mms, fs, a=0.5, b=12.0)
    n_clean = bandpass(n_mms, fs, a=0.5, b=12.0)
    e_clean = bandpass(e_mms, fs, a=0.5, b=12.0)
    horiz_clean = np.sqrt(n_clean**2 + e_clean**2)

    # Envelopes (0.5 s window)
    env_z = envelope(z_clean, fs, win_sec=0.5)
    env_h = envelope(horiz_clean, fs, win_sec=0.5)

    # Arrival times t_P and t_S
    p_level = 0.05 * np.max(env_z)
    p_idx = int(np.argmax(env_z > p_level))
    t_P = float(t[p_idx])

    s_level = 0.10 * np.max(env_h)
    after_mask = t >= (t_P + 2.0)
    after_indices = np.where(after_mask)[0]
    s_rel_idx = int(np.argmax(env_h[after_mask] > s_level))
    s_idx = int(after_indices[s_rel_idx])
    t_S = float(t[s_idx])

    # Distance R (km) and origin time
    v_P = 6.2
    v_S = 3.6
    R_km = float((t_S - t_P) / ((1.0 / v_S) - (1.0 / v_P)))
    t_origin = float(t_P - (R_km / v_P))

    # PGV (cm/s) from raw horizontal motion and magnitude M
    horiz_raw_cms = np.sqrt(n_raw**2 + e_raw**2) * 100.0
    pgv_cms = float(np.max(horiz_raw_cms))
    M = float((np.log10(pgv_cms) + np.log10(R_km) + 1.0) / 0.55)

    return {
        "record": Path(csv_file).stem,
        "t_P_s": t_P,
        "t_S_s": t_S,
        "t_origin_s": t_origin,
        "R_km": R_km,
        "PGV_cm_s": pgv_cms,
        "M": M,
    }


def main():
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [
        Path("data/seismogram_A.csv"),
        Path("data/seismogram_B.csv"),
    ]
    rows = [analyse(rec) for rec in records]
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "events.csv", index=False)


if __name__ == "__main__":
    main()