#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output" / "mplconfig"))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from plot_style import apply_mnras_style, get_single_column_size


BASE_DIR = ROOT / "output" / "inv_compton_tau_scan_eps1e-5_theta0p1_n50000_tau100"
BEAM_CSV = BASE_DIR / "beam_tau_scan_summary.csv"
LAMBERT_CSV = BASE_DIR / "lambert_tau_scan_summary.csv"
PAIR_BALANCE_CSV = ROOT / "pair_balance" / "data" / "ps96_slab_pair_line_reflection_compare_theta_0.02_1.0_log50.csv"
PNG_PATH = BASE_DIR / "eta_vs_tau_beam_lambert.png"
PDF_PATH = BASE_DIR / "eta_vs_tau_beam_lambert.pdf"


def load_eta_series(path: Path) -> tuple[list[float], list[float]]:
    tau_values: list[float] = []
    eta_values: list[float] = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tau = float(row["slab_tau"])
            escaped_up_scattered = float(row["escaped_up_scattered"])
            escaped_down_scattered = float(row["escaped_down_scattered"])
            mean_up_energy = float(row["mean_up_scattered_energy_mec2"])
            mean_down_energy = float(row["mean_down_scattered_energy_mec2"])

            upward_luminosity = escaped_up_scattered * mean_up_energy
            downward_luminosity = escaped_down_scattered * mean_down_energy
            total_luminosity = downward_luminosity + upward_luminosity
            eta = downward_luminosity / total_luminosity

            tau_values.append(tau)
            eta_values.append(eta)

    return tau_values, eta_values


def load_pair_balance_eta_series(path: Path) -> tuple[list[float], list[float]]:
    tau_values: list[float] = []
    eta_values: list[float] = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tau = float(row["tau_T_reflect"])
            downward_flux = float(row["downward_flux_model"])
            upward_flux = float(row["observed_total_up_flux_reflect_model"])
            if upward_flux <= 0.0:
                continue
            tau_values.append(tau)
            total_flux = downward_flux + upward_flux
            eta_values.append(downward_flux / total_flux)

    return tau_values, eta_values


def main() -> None:
    apply_mnras_style()

    beam_tau, beam_eta = load_eta_series(BEAM_CSV)
    lambert_tau, lambert_eta = load_eta_series(LAMBERT_CSV)
    pair_tau, pair_eta = load_pair_balance_eta_series(PAIR_BALANCE_CSV)

    fig, ax = plt.subplots(figsize=get_single_column_size(row_height_scale=1.15))
    ax.plot(
        beam_tau,
        beam_eta,
        color="#1f77b4",
        marker="o",
        markersize=3.0,
        linewidth=1.6,
        label="Beam injection",
    )
    ax.plot(
        lambert_tau,
        lambert_eta,
        color="#d95f02",
        marker="s",
        markersize=3.0,
        linewidth=1.6,
        label="Lambert injection",
    )
    ax.plot(
        pair_tau,
        pair_eta,
        color="#2a9d8f",
        linestyle="--",
        linewidth=1.6,
        label="Pair-balance reflect model",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"Optical depth $\tau$")
    ax.set_ylabel(r"$\eta = L_d / (L_d + L_c)$")
    ax.set_title(r"Escape-energy fraction: $\epsilon=10^{-5}$, $\theta=0.1$")
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
