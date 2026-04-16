#!/usr/bin/env python3
"""Slab tau-kTe scans with explicit intrinsic-disk dissipation ratio d.

This module keeps the COMPPS transfer solve from ``scanner.py`` but updates the
``d_ratio > 0`` treatment so the extra cold-disk dissipation enters through the
physical seed-photon normalization rather than only through an amplification
closure.

Model used here
---------------
For fixed ``(theta, tau_T)``, COMPPS is solved once for a unit Lambertian
blackbody source at the slab bottom.  Because the transfer step is linear in
the source normalization, that unit-source solution can be rescaled to any
physical soft compactness.

We define

    d = l_disk,int / l_diss

with ``l_diss`` the coronal dissipation compactness.  The physical seed
compactness entering the corona is then

    l_s / l_diss = d + (1 - a) * eta * (l_c / l_diss),

where ``a`` is the fixed albedo and ``eta`` is the downward fraction of the
Comptonized luminosity returned by COMPPS.

For the COMPPS unit-seed solution,

    A_model = l_c / l_s
    p_sc    = scattered fraction of the seed field.

The closure used here is

    l_c / l_diss = (1 + d * p_sc) / (1 - p_sc * eta)
    l_s / l_diss = d + (1 - a) * eta * (l_c / l_diss)
                 = [d + (1 - a) * eta - a * d * p_sc * eta] / (1 - p_sc * eta)

and therefore

    A_required = (l_c / l_diss) / (l_s / l_diss)
               = (1 + d * p_sc) / [d + (1 - a) * eta - a * d * p_sc * eta].

This reduces to the passive-disk closure when ``d = 0`` while also allowing
the extra intrinsic-disk photons to rescale the exported COMPPS internal field
and the pair-production kernel self-consistently.
"""

from __future__ import annotations

import csv
import math
import os
import pathlib
from dataclasses import dataclass

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_pair_balance_dratio")

try:
    from pair_balance.scanner import (
        CLIGHT,
        KTE_TO_THETA,
        MEC2_ERG,
        SIGMA_T,
        SLAB_HEIGHT_CM,
        ComppsSlabSolver,
        ScanConfig,
        logspace,
    )
except ModuleNotFoundError:
    from scanner import (
        CLIGHT,
        KTE_TO_THETA,
        MEC2_ERG,
        SIGMA_T,
        SLAB_HEIGHT_CM,
        ComppsSlabSolver,
        ScanConfig,
        logspace,
    )


BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
DATA_DIR = BASE / "data"
ROOT_OUTPUT_DIR = ROOT / "output"

OUTPUT_CSV = DATA_DIR / "ps96_slab_dratio_tau_kTe_theta_0.02_1.0_log40.csv"
OUTPUT_TAU_KTE_PNG = ROOT_OUTPUT_DIR / "ps96_slab_dratio_tau_kTe_family.png"
OUTPUT_KTE_LDISS_PNG = ROOT_OUTPUT_DIR / "ps96_slab_dratio_kTe_ldiss_family.png"
OUTPUT_TAU_LDISS_PNG = ROOT_OUTPUT_DIR / "ps96_slab_dratio_tau_ldiss_family.png"
D_RATIO_VALUES = (0.0, 0.25, 1.0, 3.0)


@dataclass(frozen=True)
class DRatioScanConfig(ScanConfig):
    d_ratio: float = 0.0
    theta_min: float = 0.02
    theta_max: float = 1.0
    n_samples: int = 40
    tau_min: float = 0.005
    tau_bisect_iterations: int = 32
    continuation_expand_steps: int = 8
    global_tau_samples: int = 18


class DRatioSlabSolver(ComppsSlabSolver):
    def __init__(self, config: DRatioScanConfig):
        super().__init__(config)
        self.config = config

    @staticmethod
    def _safe_positive(value: float, floor: float = 1.0e-12) -> float:
        return max(value, floor)

    def compactness_terms(self, state) -> tuple[float, float, float, float]:
        transport_denom = self._safe_positive(1.0 - state.p_sc * state.eta)
        lc_over_ldiss_model = (1.0 + self.config.d_ratio * state.p_sc) / transport_denom
        reprocessed_seed_over_ldiss = (1.0 - self.config.albedo) * state.eta * lc_over_ldiss_model
        intrinsic_seed_over_ldiss = self.config.d_ratio
        ls_over_ldiss_model = intrinsic_seed_over_ldiss + reprocessed_seed_over_ldiss
        return (
            ls_over_ldiss_model,
            lc_over_ldiss_model,
            intrinsic_seed_over_ldiss,
            reprocessed_seed_over_ldiss,
        )

    def amplification_required(self, state) -> float:
        feedback = (1.0 - self.config.albedo) * state.eta
        disk_term = self.config.d_ratio * (1.0 - self.config.albedo * state.eta * state.p_sc)
        return (1.0 + self.config.d_ratio * state.p_sc) / self._safe_positive(feedback + disk_term)

    def pair_production_rate_per_ldiss2(self, state, ls_over_ldiss: float) -> float:
        flux_scale = ls_over_ldiss * MEC2_ERG * CLIGHT / (SIGMA_T * SLAB_HEIGHT_CM * state.seed_flux_model)
        field_physical = state.internal_field_model * flux_scale
        kernel = self._ensure_kernel(state)
        return kernel.pair_production_rate(field_physical, state.tau_grid)

    def energy_residual(self, theta: float, tau_t: float) -> float:
        state = self.run_state(theta, tau_t)
        return math.log(state.amplification_model / self._safe_positive(self.amplification_required(state)))

    def solve_point(self, theta: float, guess_tau: float | None) -> dict[str, float | str]:
        tau_t, root_method = self.find_tau_root(theta, guess_tau)
        state = self.run_state(theta, tau_t)
        ls_over_ldiss, lc_over_ldiss, intrinsic_seed, reprocessed_seed = self.compactness_terms(state)
        a_required = self.amplification_required(state)
        prod_coeff = self.pair_production_rate_per_ldiss2(state, ls_over_ldiss)
        ann_rate = self.pair_annihilation_rate(theta, tau_t)
        ldiss = math.sqrt(ann_rate / self._safe_positive(prod_coeff))
        return {
            "d_ratio": self.config.d_ratio,
            "theta": theta,
            "kTe_keV": theta / KTE_TO_THETA,
            "tau_T": tau_t,
            "l_diss_local": ldiss,
            "eta": state.eta,
            "p_sc": state.p_sc,
            "A_model": state.amplification_model,
            "A_required": a_required,
            "l_s_over_l_diss": ls_over_ldiss,
            "l_c_over_l_diss": lc_over_ldiss,
            "intrinsic_seed_over_l_diss": intrinsic_seed,
            "reprocessed_seed_over_l_diss": reprocessed_seed,
            "pair_production_rate_unit_ldiss2": prod_coeff,
            "pair_annihilation_rate": ann_rate,
            "energy_log_residual": self.energy_residual(theta, tau_t),
            "root_method": root_method,
        }


def scan_single_dratio(config: DRatioScanConfig) -> list[dict[str, float | str]]:
    solver = DRatioSlabSolver(config)
    rows: list[dict[str, float | str]] = []
    previous_tau: float | None = None

    for theta in logspace(config.theta_min, config.theta_max, config.n_samples):
        row = solver.solve_point(theta, previous_tau)
        rows.append(row)
        previous_tau = float(row["tau_T"])

    return rows


def write_rows(rows: list[dict[str, float | str]], output_csv: pathlib.Path) -> None:
    output_csv.parent.mkdir(exist_ok=True)
    fieldnames = [
        "d_ratio",
        "theta",
        "kTe_keV",
        "tau_T",
        "l_diss_local",
        "eta",
        "p_sc",
        "A_model",
        "A_required",
        "l_s_over_l_diss",
        "l_c_over_l_diss",
        "intrinsic_seed_over_l_diss",
        "reprocessed_seed_over_l_diss",
        "pair_production_rate_unit_ldiss2",
        "pair_annihilation_rate",
        "energy_log_residual",
        "root_method",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_plot_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "axes.facecolor": "#fbfbf8",
            "figure.facecolor": "white",
            "legend.frameon": False,
            "font.size": 11,
        }
    )


def plot_family(
    rows: list[dict[str, float | str]],
    output_png: pathlib.Path,
    *,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    _prepare_plot_style()
    import matplotlib.pyplot as plt

    output_png.parent.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = ["#1f77b4", "#2ca02c", "#d95f02", "#9467bd", "#8c564b"]

    d_values = sorted({float(row["d_ratio"]) for row in rows})
    for color, d_value in zip(palette, d_values):
        curve = [row for row in rows if float(row["d_ratio"]) == d_value]
        curve.sort(key=lambda row: float(row[x_key]))
        ax.plot(
            [float(row[x_key]) for row in curve],
            [float(row[y_key]) for row in curve],
            color=color,
            lw=2.2,
            label=f"d = {d_value:g}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    all_rows: list[dict[str, float | str]] = []
    for d_ratio in D_RATIO_VALUES:
        config = DRatioScanConfig(d_ratio=d_ratio)
        all_rows.extend(scan_single_dratio(config))

    write_rows(all_rows, OUTPUT_CSV)
    plot_family(
        all_rows,
        OUTPUT_TAU_KTE_PNG,
        x_key="kTe_keV",
        y_key="tau_T",
        x_label=r"$kT_{\rm e}$ (keV)",
        y_label=r"$\tau_{\rm T}$",
        title="Slab Pair-Balance Tau-kTe Curves for Fixed Albedo",
    )
    plot_family(
        all_rows,
        OUTPUT_KTE_LDISS_PNG,
        x_key="kTe_keV",
        y_key="l_diss_local",
        x_label=r"$kT_{\rm e}$ (keV)",
        y_label=r"$l_{\rm diss}$",
        title="Slab Pair-Balance kTe-l_diss Curves for Fixed Albedo",
    )
    plot_family(
        all_rows,
        OUTPUT_TAU_LDISS_PNG,
        x_key="tau_T",
        y_key="l_diss_local",
        x_label=r"$\tau_{\rm T}$",
        y_label=r"$l_{\rm diss}$",
        title="Slab Pair-Balance Tau-l_diss Curves for Fixed Albedo",
    )
    print(OUTPUT_CSV)
    print(OUTPUT_TAU_KTE_PNG)
    print(OUTPUT_KTE_LDISS_PNG)
    print(OUTPUT_TAU_LDISS_PNG)


if __name__ == "__main__":
    main()
