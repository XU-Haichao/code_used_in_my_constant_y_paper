#!/usr/bin/env python3
"""compPSc f-g energy-balance curves with pair-balance critical compactness.

This scanner combines the f-g sandwich energy-balance closure in
``scanner_comppsc_fg.py`` with the gamma-gamma pair-production kernel used by
the compPSc pair-line scanner.  For each fixed electron temperature it first
solves the energy-balance root in tau, then rescales the compPSc internal
radiation field by the physical soft compactness implied by the same f-g
closure and solves the pair-balance compactness.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


ROOT_FOR_IMPORT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from pair_balance.scanner import CLIGHT, MEC2_ERG, SIGMA_T, SLAB_HEIGHT_CM, logspace
from pair_balance.scanner_comppsc_fg import (
    FGEnergyBalanceConfig,
    FGEnergyBalanceSolver,
    MEC2_KEV,
    amplification_required_fg,
    compactness_terms_fg,
    d_ratio_from_f,
    parse_float_list,
)
from pair_balance.scanner_comppsc_eta_albedo_grid import column_suffix


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pair_balance" / "data"
OUTPUT_DIR = ROOT / "figure"


@dataclass(frozen=True)
class FGPairBalanceConfig(FGEnergyBalanceConfig):
    kTe_min_kev: float = 10.0
    kTe_max_kev: float = 200.0
    n_samples: int = 24
    f_values: tuple[float, ...] = (0.1, 0.3, 1.0)
    g_values: tuple[float, ...] = (0.1, 0.3, 1.0)
    tau_min: float = 0.03
    tau_max: float = 10.0
    max_scatter: int = 4000
    global_tau_samples: int = 24


def safe_positive(value: float, floor: float = 1.0e-30) -> float:
    return max(float(value), floor)


def field_flux_scale_per_ldiss(ls_over_ldiss: float, seed_flux_model: float) -> float:
    return (
        float(ls_over_ldiss)
        * MEC2_ERG
        * CLIGHT
        / (SIGMA_T * SLAB_HEIGHT_CM * safe_positive(seed_flux_model))
    )


def pair_balance_ldiss(pair_production_rate_unit_ldiss2: float, pair_annihilation_rate: float) -> float:
    if pair_production_rate_unit_ldiss2 <= 0.0:
        return math.inf
    return math.sqrt(float(pair_annihilation_rate) / float(pair_production_rate_unit_ldiss2))


def output_paths(
    f_values: Sequence[float],
    g_values: Sequence[float],
    fixed_xi: float,
    max_scatter: int,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    f_slug = "_".join(column_suffix(value) for value in f_values)
    g_slug = "_".join(column_suffix(value) for value in g_values)
    xi_slug = column_suffix(fixed_xi)
    stem = f"comppsc_fg_pair_balance_xi{xi_slug}_f{f_slug}_g{g_slug}_maxsc{int(max_scatter)}"
    return (
        DATA_DIR / f"{stem}.csv",
        OUTPUT_DIR / f"{stem}.png",
        OUTPUT_DIR / f"{stem}.pdf",
    )


def kTe_values_for_scan(kTe_min_kev: float, kTe_max_kev: float, n_samples: int) -> list[float]:
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if n_samples == 1:
        return [float(kTe_min_kev)]
    return [float(value) for value in logspace(kTe_min_kev, kTe_max_kev, n_samples)]


def downsample_equilibrium_curve_rows(
    rows: Sequence[dict[str, float | int | str]],
    *,
    max_points: int,
) -> list[dict[str, float | int | str]]:
    if max_points < 1:
        raise ValueError("max_points must be positive")
    sorted_rows = sorted(rows, key=lambda row: float(row["kTe_keV"]))
    if len(sorted_rows) <= max_points:
        return list(sorted_rows)
    indices = [int(round(value)) for value in np.linspace(0, len(sorted_rows) - 1, max_points)]
    selected: list[dict[str, float | int | str]] = []
    previous_index: int | None = None
    for index in indices:
        if index == previous_index:
            continue
        selected.append(sorted_rows[index])
        previous_index = index
    return selected


def equilibrium_rows_from_csv(
    path: pathlib.Path,
    *,
    f_values: Sequence[float],
    g_values: Sequence[float],
    points_per_curve: int | None = None,
) -> list[dict[str, float]]:
    rows_by_curve: dict[tuple[float, float], list[dict[str, float]]] = {
        (float(f_corona), float(feedback_factor)): []
        for f_corona in f_values
        for feedback_factor in g_values
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            try:
                f_corona = float(raw["f_corona"])
                feedback_factor = float(raw["g_feedback"])
                kTe_keV = float(raw["kTe_keV"])
                tau_T = float(raw["tau_T"])
            except (KeyError, TypeError, ValueError):
                continue
            key = next(
                (
                    candidate
                    for candidate in rows_by_curve
                    if math.isclose(candidate[0], f_corona) and math.isclose(candidate[1], feedback_factor)
                ),
                None,
            )
            if key is None:
                continue
            if not (math.isfinite(kTe_keV) and math.isfinite(tau_T) and kTe_keV > 0.0 and tau_T > 0.0):
                continue
            rows_by_curve[key].append(
                {
                    "f_corona": float(key[0]),
                    "g_feedback": float(key[1]),
                    "kTe_keV": kTe_keV,
                    "tau_T": tau_T,
                }
            )

    missing = [key for key, rows in rows_by_curve.items() if not rows]
    if missing:
        raise ValueError(f"missing equilibrium rows for f,g combinations: {missing}")

    selected_rows: list[dict[str, float]] = []
    for key in rows_by_curve:
        curve_rows = rows_by_curve[key]
        if points_per_curve is not None:
            curve_rows = downsample_equilibrium_curve_rows(curve_rows, max_points=points_per_curve)
        else:
            curve_rows = sorted(curve_rows, key=lambda row: float(row["kTe_keV"]))
        selected_rows.extend(curve_rows)
    return selected_rows


class FGPairBalanceSolver(FGEnergyBalanceSolver):
    def pair_production_rate_per_ldiss2(self, state, ls_over_ldiss: float) -> float:
        flux_scale = field_flux_scale_per_ldiss(
            ls_over_ldiss=ls_over_ldiss,
            seed_flux_model=state.seed_flux_model,
        )
        field_physical = state.internal_field_model * flux_scale
        kernel = self.transfer._ensure_kernel(state)
        return kernel.pair_production_rate(field_physical, state.tau_grid)

    def evaluate_pair_tau(
        self,
        theta: float,
        tau_t: float,
        *,
        f_corona: float,
        feedback_factor: float,
    ) -> dict[str, float | int | bool | str]:
        state, albedo, reflected_flux, last_scatter, last_difmax = self.state_and_albedo(theta, tau_t)
        a_required = amplification_required_fg(state.eta, state.p_sc, albedo, f_corona, feedback_factor)
        terms = compactness_terms_fg(state.eta, state.p_sc, albedo, f_corona, feedback_factor)
        residual = math.log(state.amplification_model / safe_positive(a_required))
        prod_coeff = self.pair_production_rate_per_ldiss2(state, terms.l_s_over_l_c)
        ann_rate = self.transfer.pair_annihilation_rate(theta, tau_t)
        ldiss = pair_balance_ldiss(prod_coeff, ann_rate)
        converged = (
            last_scatter < self.config.max_scatter - 1
            and math.isfinite(last_difmax)
            and last_difmax <= 3.2e-3
        )
        return {
            "model": "compPSc-fg",
            "f_corona": float(f_corona),
            "g_feedback": float(feedback_factor),
            "d_ratio": d_ratio_from_f(f_corona),
            "theta": theta,
            "kTe_keV": theta * MEC2_KEV,
            "tau_T": tau_t,
            "l_diss_local": ldiss,
            "fixed_xi": self.config.fixed_xi,
            "effective_albedo": albedo,
            "eta": state.eta,
            "p_sc": state.p_sc,
            "A_model": state.amplification_model,
            "A_required": a_required,
            "l_s_over_l_diss": terms.l_s_over_l_c,
            "l_h_over_l_diss": terms.l_h_over_l_c,
            "intrinsic_seed_over_l_diss": terms.intrinsic_seed_over_l_c,
            "reprocessed_seed_over_l_diss": terms.reprocessed_seed_over_l_c,
            "returning_reflection_over_l_diss": terms.returning_reflection_over_l_c,
            "downward_flux_model": state.comp_down_flux_model,
            "reflected_flux_model_hemisphere": reflected_flux,
            "pair_production_rate_unit_ldiss2": prod_coeff,
            "pair_annihilation_rate": ann_rate,
            "energy_log_residual": residual,
            "last_scatter_order": last_scatter,
            "last_difmax": last_difmax,
            "converged": converged,
            "max_scatter": self.config.max_scatter,
            "hemisphere_mu_order": self.config.hemisphere_mu_order,
            "exact_angles": self.config.exact_angles,
        }

    def solve_pair_point(
        self,
        theta: float,
        *,
        f_corona: float,
        feedback_factor: float,
        guess_tau: float | None,
    ) -> dict[str, float | int | bool | str]:
        tau_t, root_method, root_iterations = self.find_tau_root(
            theta,
            f_corona=f_corona,
            feedback_factor=feedback_factor,
            guess_tau=guess_tau,
        )
        row = self.evaluate_pair_tau(theta, tau_t, f_corona=f_corona, feedback_factor=feedback_factor)
        row["root_method"] = root_method
        row["root_iterations"] = root_iterations
        return row


def scan_fg_pair_curves(config: FGPairBalanceConfig) -> list[dict[str, float | int | bool | str]]:
    solver = FGPairBalanceSolver(config)
    theta_values = [
        value / MEC2_KEV
        for value in kTe_values_for_scan(config.kTe_min_kev, config.kTe_max_kev, config.n_samples)
    ]
    rows: list[dict[str, float | int | bool | str]] = []
    for f_corona in config.f_values:
        for feedback_factor in config.g_values:
            previous_tau: float | None = None
            for theta in theta_values:
                row = solver.solve_pair_point(
                    theta,
                    f_corona=f_corona,
                    feedback_factor=feedback_factor,
                    guess_tau=previous_tau,
                )
                rows.append(row)
                previous_tau = float(row["tau_T"])
                print(
                    f"f={f_corona:g} g={feedback_factor:g} "
                    f"kTe={float(row['kTe_keV']):.5g} tau={float(row['tau_T']):.6g} "
                    f"ldiss={float(row['l_diss_local']):.5g} "
                    f"sc={int(row['last_scatter_order'])} dif={float(row['last_difmax']):.3g} "
                    f"conv={row['converged']}",
                    flush=True,
                )
    return rows


def scan_fg_pair_from_equilibrium_rows(
    config: FGPairBalanceConfig,
    equilibrium_rows: Sequence[dict[str, float]],
) -> list[dict[str, float | int | bool | str]]:
    solver = FGPairBalanceSolver(config)
    rows: list[dict[str, float | int | bool | str]] = []
    ordered_rows = sorted(
        equilibrium_rows,
        key=lambda row: (float(row["f_corona"]), float(row["g_feedback"]), float(row["kTe_keV"])),
    )
    for equilibrium_row in ordered_rows:
        f_corona = float(equilibrium_row["f_corona"])
        feedback_factor = float(equilibrium_row["g_feedback"])
        kTe_keV = float(equilibrium_row["kTe_keV"])
        tau_t = float(equilibrium_row["tau_T"])
        row = solver.evaluate_pair_tau(
            kTe_keV / MEC2_KEV,
            tau_t,
            f_corona=f_corona,
            feedback_factor=feedback_factor,
        )
        row["root_method"] = "input_equilibrium_curve"
        row["root_iterations"] = 0
        rows.append(row)
        print(
            f"f={f_corona:g} g={feedback_factor:g} "
            f"kTe={kTe_keV:.5g} tau={tau_t:.6g} "
            f"ldiss={float(row['l_diss_local']):.5g} "
            f"sc={int(row['last_scatter_order'])} dif={float(row['last_difmax']):.3g} "
            f"conv={row['converged']}",
            flush=True,
        )
    return rows


def write_rows(path: pathlib.Path, rows: Sequence[dict[str, float | int | bool | str]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty f-g pair-balance scan")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_pair_curves(
    png_path: pathlib.Path,
    pdf_path: pathlib.Path,
    rows: Sequence[dict[str, float | int | bool | str]],
    *,
    f_values: Sequence[float],
    g_values: Sequence[float],
    ldiss_ylim: tuple[float, float] = (10.0, 1.0e6),
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_comppsc_fg_pair")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))
    f_colors = dict(zip(f_values, plt.get_cmap("plasma")(np.linspace(0.18, 0.86, len(f_values)))))
    default_linestyles = [":", "--", "-", "-."]
    g_linestyles = {
        float(feedback_factor): default_linestyles[index % len(default_linestyles)]
        for index, feedback_factor in enumerate(g_values)
    }
    panels = (
        (axes[0], "kTe_keV", "tau_T", r"$kT_{\rm e}$ (keV)", r"$\tau_{\rm T}$"),
        (axes[1], "kTe_keV", "l_diss_local", r"$kT_{\rm e}$ (keV)", r"$l_{\rm diss}$"),
        (axes[2], "tau_T", "l_diss_local", r"$\tau_{\rm T}$", r"$l_{\rm diss}$"),
    )
    for f_corona in f_values:
        for feedback_factor in g_values:
            selected = sorted(
                [
                    row
                    for row in rows
                    if math.isclose(float(row["f_corona"]), f_corona)
                    and math.isclose(float(row["g_feedback"]), feedback_factor)
                ],
                key=lambda row: float(row["kTe_keV"]),
            )
            if not selected:
                continue
            color = f_colors[f_corona]
            linestyle = g_linestyles.get(float(feedback_factor), "-")
            for ax, x_key, y_key, _, _ in panels:
                ax.plot(
                    [float(row[x_key]) for row in selected],
                    [float(row[y_key]) for row in selected],
                    color=color,
                    linestyle=linestyle,
                    lw=1.9,
                )

    for ax, _, _, xlabel, ylabel in panels:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(False)
    axes[1].set_ylim(*ldiss_ylim)
    axes[2].set_ylim(*ldiss_ylim)

    f_handles = [Line2D([], [], color=f_colors[f_corona], lw=2.0, label=rf"$f={f_corona:g}$") for f_corona in f_values]
    g_handles = [
        Line2D([], [], color="0.2", lw=1.9, linestyle=g_linestyles[float(g)], label=rf"$g={g:g}$")
        for g in g_values
    ]
    axes[0].legend(handles=f_handles + g_handles, frameon=False, fontsize=8.0, ncol=2)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kTe-min", type=float, default=FGPairBalanceConfig.kTe_min_kev)
    parser.add_argument("--kTe-max", type=float, default=FGPairBalanceConfig.kTe_max_kev)
    parser.add_argument("--n-samples", type=int, default=FGPairBalanceConfig.n_samples)
    parser.add_argument("--f-values", nargs="*", default=None, help="Space/comma separated coronal fractions.")
    parser.add_argument("--g-values", nargs="*", default=None, help="Space/comma separated feedback factors.")
    parser.add_argument("--fixed-xi", type=float, default=FGPairBalanceConfig.fixed_xi)
    parser.add_argument("--max-scatter", type=int, default=FGPairBalanceConfig.max_scatter)
    parser.add_argument("--mu-order", type=int, default=FGPairBalanceConfig.hemisphere_mu_order)
    parser.add_argument("--tau-min", type=float, default=FGPairBalanceConfig.tau_min)
    parser.add_argument("--tau-max", type=float, default=FGPairBalanceConfig.tau_max)
    parser.add_argument("--root-tolerance", type=float, default=FGPairBalanceConfig.root_tolerance)
    parser.add_argument("--global-tau-samples", type=int, default=FGPairBalanceConfig.global_tau_samples)
    parser.add_argument("--ldiss-ymin", type=float, default=10.0)
    parser.add_argument("--ldiss-ymax", type=float, default=1.0e6)
    parser.add_argument("--output-csv", type=pathlib.Path, default=None)
    parser.add_argument("--output-png", type=pathlib.Path, default=None)
    parser.add_argument("--output-pdf", type=pathlib.Path, default=None)
    parser.add_argument(
        "--input-equilibrium-csv",
        type=pathlib.Path,
        default=None,
        help="Optional precomputed f-g energy-balance curve CSV with f_corona,g_feedback,kTe_keV,tau_T columns.",
    )
    parser.add_argument(
        "--points-per-curve",
        type=int,
        default=None,
        help="Optional number of input equilibrium points to keep per f,g curve.",
    )
    parser.add_argument("--no-exact-angles", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    f_values = parse_float_list(args.f_values, FGPairBalanceConfig.f_values)
    g_values = parse_float_list(args.g_values, FGPairBalanceConfig.g_values)
    config = FGPairBalanceConfig(
        kTe_min_kev=args.kTe_min,
        kTe_max_kev=args.kTe_max,
        n_samples=args.n_samples,
        f_values=f_values,
        g_values=g_values,
        fixed_xi=args.fixed_xi,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        max_scatter=args.max_scatter,
        exact_angles=not args.no_exact_angles,
        hemisphere_mu_order=args.mu_order,
        root_tolerance=args.root_tolerance,
        global_tau_samples=args.global_tau_samples,
    )
    if args.input_equilibrium_csv is None:
        rows = scan_fg_pair_curves(config)
    else:
        equilibrium_rows = equilibrium_rows_from_csv(
            args.input_equilibrium_csv,
            f_values=f_values,
            g_values=g_values,
            points_per_curve=args.points_per_curve,
        )
        rows = scan_fg_pair_from_equilibrium_rows(config, equilibrium_rows)
    default_csv, default_png, default_pdf = output_paths(f_values, g_values, args.fixed_xi, args.max_scatter)
    output_csv = args.output_csv or default_csv
    output_png = args.output_png or default_png
    output_pdf = args.output_pdf or default_pdf
    write_rows(output_csv, rows)
    plot_pair_curves(
        output_png,
        output_pdf,
        rows,
        f_values=f_values,
        g_values=g_values,
        ldiss_ylim=(args.ldiss_ymin, args.ldiss_ymax),
    )
    print(output_csv)
    print(output_png)
    print(output_pdf)


if __name__ == "__main__":
    main()
