#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "inv_compton"
BUILD_DIR = ROOT / "output" / "inv_compton_build"
BIN_PATH = BUILD_DIR / "inv_compton_sim"
OUT_DIR = ROOT / "output" / "inv_compton_single_scatter_theta_sweep_eps1e-5_n50000"
SUMMARY_CSV = OUT_DIR / "single_scatter_theta_sweep_summary.csv"


def make_log_grid(min_value: float, max_value: float, count: int) -> list[float]:
    if count <= 1:
        return [min_value]
    log_min = math.log(min_value)
    log_max = math.log(max_value)
    return [
        math.exp(log_min + (log_max - log_min) * i / (count - 1))
        for i in range(count)
    ]


def build_binary() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted(str(path) for path in SRC_DIR.glob("*.cpp"))
    cmd = [
        "/usr/bin/clang++",
        "-std=c++17",
        "-O3",
        "-I",
        "/Users/epiphyllum/anaconda3/include",
        *sources,
        "-L",
        "/Users/epiphyllum/anaconda3/lib",
        "-lhdf5_cpp",
        "-lhdf5",
        "-Wl,-rpath,/Users/epiphyllum/anaconda3/lib",
        "-o",
        str(BIN_PATH),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def read_key_value_csv(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["key"]: row["value"] for row in reader}


def main() -> None:
    photon_energy = 1.0e-5
    num_events = 50000
    theta_values = make_log_grid(1.0e-2, 1.0, 100)

    build_binary()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for index, theta_e in enumerate(theta_values):
        label = f"single_scatter_theta_{index:03d}"
        seed = 2026041301 + index
        cmd = [
            str(BIN_PATH),
            "--mode",
            "run",
            "--events",
            str(num_events),
            "--seed",
            str(seed),
            "--geometry",
            "none",
            "--electron-model",
            "thermal",
            "--electron-kTe",
            f"{theta_e:.16g}",
            "--photon-energy",
            f"{photon_energy:.16g}",
            "--label",
            label,
            "--output-dir",
            str(OUT_DIR),
        ]
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        summary_candidates = sorted(
            OUT_DIR.glob(f"*{label}*_run_summary.csv"),
            key=lambda path: path.stat().st_mtime,
        )
        if not summary_candidates:
            raise RuntimeError(f"Could not find summary CSV for label={label}")
        summary = read_key_value_csv(summary_candidates[-1])
        amplification = float(summary["mean_scattered_energy_mec2"]) / photon_energy

        rows.append(
            {
                "theta_e_mec2": f"{theta_e:.16g}",
                "incident_photon_energy_mec2": f"{photon_energy:.16g}",
                "events_requested": str(num_events),
                "seed": str(seed),
                "source_run_tag": summary["run_tag"],
                "mean_scattered_energy_mec2": summary["mean_scattered_energy_mec2"],
                "mean_scattered_energy_over_incident": f"{amplification:.16g}",
                "mean_energy_gain_over_incident": f"{(amplification - 1.0):.16g}",
                "mean_sampled_electron_gamma": summary["mean_sampled_electron_gamma"],
                "stddev_sampled_electron_gamma": summary["stddev_sampled_electron_gamma"],
                "mean_photon_energy_ratio_erf": summary["mean_photon_energy_ratio_erf"],
                "energy_hist_overflow_fraction": summary["energy_hist_overflow_fraction"],
            }
        )

    with SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
