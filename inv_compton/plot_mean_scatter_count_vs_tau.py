#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output" / "mplconfig"))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from plot_style import apply_mnras_style, get_single_column_size


BASE_DIR = ROOT / "output" / "inv_compton_tau_scan_eps1e-5_theta0p1_n50000_tau100"
BEAM_CSV = BASE_DIR / "beam_tau_scan_summary.csv"
LAMBERT_CSV = BASE_DIR / "lambert_tau_scan_summary.csv"
PNG_PATH = BASE_DIR / "mean_scatter_count_vs_tau_beam_lambert.png"
PDF_PATH = BASE_DIR / "mean_scatter_count_vs_tau_beam_lambert.pdf"


def load_series(path: Path) -> tuple[list[float], list[float]]:
    tau_values: list[float] = []
    mean_scatter_counts: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tau_values.append(float(row["slab_tau"]))
            mean_scatter_counts.append(float(row["mean_scatter_count"]))
    return tau_values, mean_scatter_counts


def main() -> None:
    apply_mnras_style()

    beam_tau, beam_mean = load_series(BEAM_CSV)
    lambert_tau, lambert_mean = load_series(LAMBERT_CSV)
    tau_array = np.array(beam_tau, dtype=float)
    tau_trend = tau_array
    tau2_trend = tau_array ** 2

    fig, ax = plt.subplots(figsize=get_single_column_size(row_height_scale=1.15))
    ax.plot(
        beam_tau,
        beam_mean,
        color="#1f77b4",
        marker="o",
        markersize=3.0,
        linewidth=1.6,
        label="Beam injection",
    )
    ax.plot(
        lambert_tau,
        lambert_mean,
        color="#d95f02",
        marker="s",
        markersize=3.0,
        linewidth=1.6,
        label="Lambert injection",
    )
    ax.plot(
        tau_array,
        tau_trend,
        color="#2a9d8f",
        linestyle="--",
        linewidth=1.4,
        label=r"$\propto \tau$",
    )
    ax.plot(
        tau_array,
        tau2_trend,
        color="#6a4c93",
        linestyle=":",
        linewidth=1.6,
        label=r"$\propto \tau^2$",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(5.0e-3, 70.0)
    ax.set_xlabel(r"Optical depth $\tau$")
    ax.set_ylabel("Mean scatter count")
    ax.set_title(r"Inverse Compton slab (log-log): $\epsilon=10^{-5}$, $\theta=0.1$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.55)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=300)
    fig.savefig(PDF_PATH)
    plt.close(fig)

    print(f"Wrote: {PNG_PATH}")
    print(f"Wrote: {PDF_PATH}")


if __name__ == "__main__":
    main()
