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
from scipy.special import kv

from plot_style import apply_mnras_style, get_single_column_size


BASE_DIR = ROOT / "output" / "inv_compton_single_scatter_theta_sweep_eps1e-5_n50000"
SUMMARY_CSV = BASE_DIR / "single_scatter_theta_sweep_summary.csv"
PNG_PATH = BASE_DIR / "single_scatter_gain_vs_theta_loglog.png"
PDF_PATH = BASE_DIR / "single_scatter_gain_vs_theta_loglog.pdf"


def main() -> None:
    apply_mnras_style()

    theta_values: list[float] = []
    gain_values: list[float] = []

    with SUMMARY_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            theta_values.append(float(row["theta_e_mec2"]))
            gain_values.append(float(row["mean_energy_gain_over_incident"]))

    theta_array = np.array(theta_values, dtype=float)
    inverse_theta = 1.0 / theta_array
    bessel_gain = 4.0 * theta_array * kv(1, inverse_theta) / kv(2, inverse_theta)
    bessel_gain += 16.0 * theta_array * theta_array
    thomson_gain = 4.0 * theta_array + 16.0 * theta_array * theta_array

    fig, ax = plt.subplots(figsize=get_single_column_size(row_height_scale=1.15))
    ax.plot(
        theta_values,
        gain_values,
        color="#1f77b4",
        marker="o",
        markersize=2.5,
        linewidth=1.5,
        label="Monte Carlo",
    )
    ax.plot(
        theta_array,
        bessel_gain,
        color="#d95f02",
        linestyle="--",
        linewidth=1.6,
        label=r"$4\theta K_1(1/\theta)/K_2(1/\theta) + 16\theta^2$",
    )
    ax.plot(
        theta_array,
        thomson_gain,
        color="#2a9d8f",
        linestyle="-.",
        linewidth=1.6,
        label=r"$4\theta + 16\theta^2$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Electron temperature $\theta$")
    ax.set_ylabel(r"$\langle \epsilon \rangle / \epsilon_0 - 1$")
    ax.set_title(r"Single-scatter energy gain, $\epsilon_0 = 10^{-5}$")
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
