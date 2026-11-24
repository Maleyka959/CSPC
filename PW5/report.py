from pathlib import Path
import pandas as pd


def main():
    events_path = Path("results/events.csv")
    report_path = Path("results/report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(events_path)

    lines = ["Earthquake Summary Report", "=" * 24, ""]
    for _, row in df.iterrows():
        lines.append(
            f"{row['record']}: t_P = {row['t_P_s']:.2f} s, t_S = {row['t_S_s']:.2f} s, "
            f"origin = {row['t_origin_s']:.2f} s, distance R = {row['R_km']:.2f} km, "
            f"PGV = {row['PGV_cm_s']:.4f} cm/s, magnitude M = {row['M']:.2f}."
        )

    closest_row = df.loc[df["R_km"].idxmin()]
    strongest_row = df.loc[df["M"].idxmax()]

    lines.append("")
    lines.append(
        f"Nearest earthquake: {closest_row['record']} (R = {closest_row['R_km']:.2f} km)."
    )
    lines.append(
        f"Strongest earthquake: {strongest_row['record']} (M = {strongest_row['M']:.2f})."
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()