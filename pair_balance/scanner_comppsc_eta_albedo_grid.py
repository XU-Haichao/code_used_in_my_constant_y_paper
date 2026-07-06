#!/usr/bin/env python3
"""Scan compPSc eta and disk-response albedo on a kTe-tau grid."""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import sys
from collections.abc import Sequence

import numpy as np


ROOT_FOR_IMPORT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from pair_balance.scanner_comppsc import ComppscScanConfig, ComppscSlabSolver
from pair_balance.scanner_comppsc_ireflect import (
    band_energy_flux,
    blackbody_surface_flux,
    ev_to_kelvin,
    ionization_parameter,
)
from pair_balance.scanner_reflect import (
    IonizedReflectionConfig,
    IonizedReflectionKernel,
    NeutralReflectionKernel,
    ReflectionConfig,
)


MEC2_KEV = 511.0
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pair_balance" / "data"
OUTPUT_DIR = ROOT / "output"

DEFAULT_KTE_GRID = [10.0, 15.0, 20.0, 30.0, 50.0, 80.0, 120.0, 200.0]
DEFAULT_TAU_GRID = [0.03, 0.05, 0.08, 0.13, 0.2, 0.32, 0.5, 0.8, 1.25, 2.0, 3.2, 5.0, 8.0, 10.0]
DEFAULT_DENSITIES = [1.0e13, 1.0e15]

LONG_CSV = DATA_DIR / "comppsc_eta_albedo_kTe_tau_grid.csv"
ETA_MATRIX_CSV = DATA_DIR / "comppsc_eta_kTe_tau_interpolation_matrix.csv"
ETA_VALID_MASK_CSV = DATA_DIR / "comppsc_eta_kTe_tau_valid_mask.csv"
A_MODEL_MATRIX_CSV = DATA_DIR / "comppsc_A_model_kTe_tau_interpolation_matrix.csv"
P_SC_MATRIX_CSV = DATA_DIR / "comppsc_p_sc_kTe_tau_interpolation_matrix.csv"
ETA_SUMMARY_CSV = DATA_DIR / "comppsc_eta_tau_dependence_summary.csv"
ALBEDO_SUMMARY_CSV = DATA_DIR / "comppsc_albedo_reflect_ireflect_summary.csv"
ETA_PNG = OUTPUT_DIR / "comppsc_eta_kTe_tau_dependence.png"
ALBEDO_PNG = OUTPUT_DIR / "comppsc_effective_albedo_reflect_ireflect_grid.png"


def column_suffix(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text


def fixed_xi_column(xi: float) -> str:
    return f"effective_albedo_ireflect_xi{column_suffix(xi)}"


def fixed_xi_output_paths(xi: float) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    slug = f"xi{column_suffix(xi)}"
    return (
        DATA_DIR / f"comppsc_albedo_ireflect_{slug}_kTe_tau_grid.csv",
        DATA_DIR / f"comppsc_albedo_ireflect_{slug}_kTe_tau_interpolation_matrix.csv",
        DATA_DIR / f"comppsc_albedo_ireflect_{slug}_kTe_tau_valid_mask.csv",
        OUTPUT_DIR / f"comppsc_albedo_ireflect_{slug}_kTe_tau.png",
    )


def geometric_edges(values: Sequence[float]) -> np.ndarray:
    centers = np.asarray(values, dtype=float)
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("grid centers must be a non-empty one-dimensional sequence")
    if np.any(centers <= 0.0) or np.any(np.diff(centers) <= 0.0):
        raise ValueError("grid centers must be positive and strictly increasing")
    if centers.size == 1:
        return np.array([centers[0] / math.sqrt(2.0), centers[0] * math.sqrt(2.0)])
    log_centers = np.log(centers)
    log_edges = np.empty(centers.size + 1, dtype=float)
    log_edges[1:-1] = 0.5 * (log_centers[:-1] + log_centers[1:])
    log_edges[0] = 2.0 * log_centers[0] - log_edges[1]
    log_edges[-1] = 2.0 * log_centers[-1] - log_edges[-2]
    return np.exp(log_edges)


def rows_to_matrix(
    rows: Sequence[dict[str, float | int | str | bool]],
    *,
    value_key: str,
    kTe_values: Sequence[float],
    tau_values: Sequence[float],
) -> np.ndarray:
    matrix = np.full((len(tau_values), len(kTe_values)), np.nan, dtype=float)
    index = {
        (round(float(row["kTe_keV"]), 10), round(float(row["tau_T"]), 10)): row
        for row in rows
    }
    for i_tau, tau_t in enumerate(tau_values):
        for i_kte, kTe_kev in enumerate(kTe_values):
            row = index.get((round(float(kTe_kev), 10), round(float(tau_t), 10)))
            if row is not None:
                if value_key == "converged":
                    matrix[i_tau, i_kte] = 1.0 if row_is_converged(row) else 0.0
                else:
                    matrix[i_tau, i_kte] = float(row[value_key])
    return matrix


def row_is_converged(row: dict[str, float | int | str | bool]) -> bool:
    value = row["converged"]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes"}
    return bool(value)


def rows_to_converged_matrix(
    rows: Sequence[dict[str, float | int | str | bool]],
    *,
    value_key: str,
    kTe_values: Sequence[float],
    tau_values: Sequence[float],
) -> np.ndarray:
    converged_rows = [row for row in rows if row_is_converged(row)]
    return rows_to_matrix(converged_rows, value_key=value_key, kTe_values=kTe_values, tau_values=tau_values)


def write_matrix_csv(
    path: pathlib.Path,
    matrix: np.ndarray,
    *,
    kTe_values: Sequence[float],
    tau_values: Sequence[float],
    value_prefix: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tau_T"] + [f"{value_prefix}_kTe_{column_suffix(kTe)}" for kTe in kTe_values]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for tau_t, values in zip(tau_values, matrix):
            row = {"tau_T": tau_t}
            for kTe_kev, value in zip(kTe_values, values):
                row[f"{value_prefix}_kTe_{column_suffix(kTe_kev)}"] = value
            writer.writerow(row)


def write_rows(path: pathlib.Path, rows: Sequence[dict[str, float | int | str | bool]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_eta_dependence(
    rows: Sequence[dict[str, float | int | str | bool]],
    tau_values: Sequence[float],
) -> list[dict[str, float | int]]:
    summary: list[dict[str, float | int]] = []
    for tau_t in tau_values:
        selected = [
            row
            for row in rows
            if round(float(row["tau_T"]), 10) == round(float(tau_t), 10)
            and row_is_converged(row)
            and math.isfinite(float(row["eta"]))
        ]
        if selected:
            eta_values = np.array([float(row["eta"]) for row in selected], dtype=float)
            mean_eta = float(np.mean(eta_values))
            min_eta = float(np.min(eta_values))
            max_eta = float(np.max(eta_values))
            rel_range = (max_eta - min_eta) / max(mean_eta, 1.0e-30)
        else:
            mean_eta = math.nan
            min_eta = math.nan
            max_eta = math.nan
            rel_range = math.nan
        summary.append(
            {
                "tau_T": float(tau_t),
                "n_converged_kTe": len(selected),
                "eta_mean_over_kTe": mean_eta,
                "eta_min_over_kTe": min_eta,
                "eta_max_over_kTe": max_eta,
                "eta_absolute_range_over_kTe": max_eta - min_eta if selected else math.nan,
                "eta_relative_range_over_kTe": rel_range,
            }
        )
    return summary


def summarize_albedo(rows: Sequence[dict[str, float | int | str | bool]]) -> list[dict[str, float | int]]:
    converged = [row for row in rows if row_is_converged(row)]
    keys = ["effective_albedo_reflect", "effective_albedo_ireflect_n1e13", "effective_albedo_ireflect_n1e15"]
    summary: list[dict[str, float | int]] = []
    for key in keys:
        values = np.array([float(row[key]) for row in converged if float(row[key]) > 0.0], dtype=float)
        if values.size == 0:
            continue
        summary.append(
            {
                "quantity": key,
                "n_points": int(values.size),
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
        )

    reflect_values = np.array([float(row["effective_albedo_reflect"]) for row in converged], dtype=float)
    for density_label in ["n1e13", "n1e15"]:
        ion_values = np.array([float(row[f"effective_albedo_ireflect_{density_label}"]) for row in converged], dtype=float)
        ratio = ion_values / np.maximum(reflect_values, 1.0e-30)
        summary.append(
            {
                "quantity": f"ireflect_{density_label}_over_reflect",
                "n_points": int(ratio.size),
                "min": float(np.min(ratio)),
                "median": float(np.median(ratio)),
                "max": float(np.max(ratio)),
            }
        )
    return summary


def effective_albedo(hemisphere_flux: float, downward_flux: float) -> float:
    return min(max(float(hemisphere_flux) / max(float(downward_flux), 1.0e-30), 0.0), 0.999999)


def scan_grid(
    *,
    kTe_values: Sequence[float],
    tau_values: Sequence[float],
    densities_cm3: Sequence[float],
    max_scatter: int,
    convergence_tolerance: float,
    tbb_kev: float,
    hemisphere_mu_order: int,
    observer_mu: float,
    exact_angles: bool,
) -> list[dict[str, float | int | str | bool]]:
    disk_temperature_k = ev_to_kelvin(1000.0 * tbb_kev)
    seed_surface_flux_cgs = blackbody_surface_flux(disk_temperature_k)
    transfer = ComppscSlabSolver(
        ComppscScanConfig(
            theta_min=min(kTe_values) / MEC2_KEV,
            theta_max=max(kTe_values) / MEC2_KEV,
            n_samples=1,
            tbb_kev=tbb_kev,
            max_scatter=max_scatter,
            exact_angles=exact_angles,
            observer_mu=observer_mu,
            tau_min=min(tau_values),
            tau_max=max(tau_values),
        )
    )
    neutral_reflector = NeutralReflectionKernel(
        ReflectionConfig(
            hemisphere_mu_order=hemisphere_mu_order,
            observer_mu=observer_mu,
        )
    )

    rows: list[dict[str, float | int | str | bool]] = []
    fieldnames: list[str] | None = None
    LONG_CSV.parent.mkdir(parents=True, exist_ok=True)
    with LONG_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer: csv.DictWriter | None = None
        for kTe_kev in kTe_values:
            theta = float(kTe_kev) / MEC2_KEV
            for tau_t in tau_values:
                state = transfer.run_state(theta, float(tau_t))
                last_scatter = int(getattr(transfer.module.msiterstat, "lastisc", -1))
                last_difmax = float(getattr(transfer.module.msiterstat, "lastdifmax", math.nan))
                converged = last_scatter < max_scatter - 1 and math.isfinite(last_difmax) and last_difmax <= convergence_tolerance

                ionizing_flux_model = band_energy_flux(
                    state.x_grid,
                    state.x_weights,
                    state.comp_down_spectrum_model,
                    energy_min_kev=0.005,
                    energy_max_kev=20.0,
                )
                ionizing_flux_cgs = ionizing_flux_model / max(state.seed_flux_model, 1.0e-30) * seed_surface_flux_cgs

                _, _, _, neutral_hemisphere_flux = neutral_reflector.hemisphere_response(
                    state.x_grid,
                    state.x_weights,
                    state.comp_down_spectrum_model,
                    observer_mu,
                )

                row: dict[str, float | int | str | bool] = {
                    "kTe_keV": float(kTe_kev),
                    "theta": theta,
                    "tau_T": float(tau_t),
                    "eta": state.eta,
                    "p_sc": state.p_sc,
                    "A_model": state.amplification_model,
                    "seed_flux_model": state.seed_flux_model,
                    "downward_flux_model": state.comp_down_flux_model,
                    "ionizing_flux_model_5eV_20keV": ionizing_flux_model,
                    "ionizing_flux_cgs_5eV_20keV": ionizing_flux_cgs,
                    "effective_albedo_reflect": effective_albedo(neutral_hemisphere_flux, state.comp_down_flux_model),
                    "last_scatter_order": last_scatter,
                    "last_difmax": last_difmax,
                    "converged": converged,
                    "max_scatter": max_scatter,
                    "convergence_tolerance": convergence_tolerance,
                    "tbb_keV": tbb_kev,
                    "disk_temperature_K": disk_temperature_k,
                    "seed_surface_flux_cgs": seed_surface_flux_cgs,
                    "hemisphere_mu_order": hemisphere_mu_order,
                    "observer_mu": observer_mu,
                    "exact_angles": exact_angles,
                }

                for density_cm3 in densities_cm3:
                    xi = ionization_parameter(ionizing_flux_cgs, density_cm3=float(density_cm3))
                    ion_reflector = IonizedReflectionKernel(
                        IonizedReflectionConfig(
                            disk_temperature_k=disk_temperature_k,
                            ionization_parameter=xi,
                            hemisphere_mu_order=hemisphere_mu_order,
                        )
                    )
                    _, _, _, ion_hemisphere_flux = ion_reflector.hemisphere_response(
                        state.x_grid,
                        state.x_weights,
                        state.comp_down_spectrum_model,
                        observer_mu,
                    )
                    suffix = f"n1e{int(round(math.log10(float(density_cm3))))}"
                    row[f"xi_{suffix}"] = xi
                    row[f"effective_albedo_ireflect_{suffix}"] = effective_albedo(
                        ion_hemisphere_flux,
                        state.comp_down_flux_model,
                    )

                if writer is None:
                    fieldnames = list(row)
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                print(
                    f"kTe={kTe_kev:g} tau={tau_t:g} eta={state.eta:.6g} "
                    f"a_ref={row['effective_albedo_reflect']:.3g} "
                    f"sc={last_scatter} dif={last_difmax:.3g} conv={converged}",
                    flush=True,
                )

    if fieldnames is None:
        raise RuntimeError("grid scan produced no rows")
    return rows


def scan_fixed_xi_grid(
    *,
    kTe_values: Sequence[float],
    tau_values: Sequence[float],
    fixed_xi: float,
    max_scatter: int,
    convergence_tolerance: float,
    tbb_kev: float,
    hemisphere_mu_order: int,
    observer_mu: float,
    exact_angles: bool,
) -> list[dict[str, float | int | str | bool]]:
    disk_temperature_k = ev_to_kelvin(1000.0 * tbb_kev)
    transfer = ComppscSlabSolver(
        ComppscScanConfig(
            theta_min=min(kTe_values) / MEC2_KEV,
            theta_max=max(kTe_values) / MEC2_KEV,
            n_samples=1,
            tbb_kev=tbb_kev,
            max_scatter=max_scatter,
            exact_angles=exact_angles,
            observer_mu=observer_mu,
            tau_min=min(tau_values),
            tau_max=max(tau_values),
        )
    )
    reflector = IonizedReflectionKernel(
        IonizedReflectionConfig(
            disk_temperature_k=disk_temperature_k,
            ionization_parameter=float(fixed_xi),
            hemisphere_mu_order=hemisphere_mu_order,
        )
    )

    value_key = fixed_xi_column(fixed_xi)
    long_csv, _, _, _ = fixed_xi_output_paths(fixed_xi)
    long_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str | bool]] = []
    with long_csv.open("w", encoding="utf-8", newline="") as handle:
        writer: csv.DictWriter | None = None
        for kTe_kev in kTe_values:
            theta = float(kTe_kev) / MEC2_KEV
            for tau_t in tau_values:
                state = transfer.run_state(theta, float(tau_t))
                last_scatter = int(getattr(transfer.module.msiterstat, "lastisc", -1))
                last_difmax = float(getattr(transfer.module.msiterstat, "lastdifmax", math.nan))
                converged = last_scatter < max_scatter - 1 and math.isfinite(last_difmax) and last_difmax <= convergence_tolerance
                _, _, _, hemisphere_flux = reflector.hemisphere_response(
                    state.x_grid,
                    state.x_weights,
                    state.comp_down_spectrum_model,
                    observer_mu,
                )
                row: dict[str, float | int | str | bool] = {
                    "kTe_keV": float(kTe_kev),
                    "theta": theta,
                    "tau_T": float(tau_t),
                    "fixed_xi": float(fixed_xi),
                    "eta": state.eta,
                    "p_sc": state.p_sc,
                    "A_model": state.amplification_model,
                    "downward_flux_model": state.comp_down_flux_model,
                    value_key: effective_albedo(hemisphere_flux, state.comp_down_flux_model),
                    "last_scatter_order": last_scatter,
                    "last_difmax": last_difmax,
                    "converged": converged,
                    "max_scatter": max_scatter,
                    "convergence_tolerance": convergence_tolerance,
                    "tbb_keV": tbb_kev,
                    "disk_temperature_K": disk_temperature_k,
                    "hemisphere_mu_order": hemisphere_mu_order,
                    "observer_mu": observer_mu,
                    "exact_angles": exact_angles,
                }
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                print(
                    f"kTe={kTe_kev:g} tau={tau_t:g} "
                    f"a_xi{fixed_xi:g}={row[value_key]:.6g} "
                    f"sc={last_scatter} dif={last_difmax:.3g} conv={converged}",
                    flush=True,
                )
    return rows


def plot_eta_dependence(
    rows: Sequence[dict[str, float | int | str | bool]],
    *,
    kTe_values: Sequence[float],
    tau_values: Sequence[float],
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_comppsc_eta_grid")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = summarize_eta_dependence(rows, tau_values)
    cmap = plt.get_cmap("viridis")
    fig, (ax_eta, ax_range) = plt.subplots(1, 2, figsize=(12.5, 5.1))
    for idx, kTe_kev in enumerate(kTe_values):
        color = cmap(idx / max(len(kTe_values) - 1, 1))
        selected = sorted(
            [row for row in rows if round(float(row["kTe_keV"]), 10) == round(float(kTe_kev), 10)],
            key=lambda row: float(row["tau_T"]),
        )
        tau = np.array([float(row["tau_T"]) for row in selected], dtype=float)
        eta = np.array([float(row["eta"]) for row in selected], dtype=float)
        good = np.array([row_is_converged(row) for row in selected], dtype=bool)
        ax_eta.plot(tau[good], eta[good], color=color, lw=1.8, marker="o", markersize=3.6, label=f"{kTe_kev:g} keV")
        if np.any(~good):
            ax_eta.scatter(tau[~good], eta[~good], color=color, marker="x", s=34, linewidths=1.5)

    ax_eta.set_xscale("log")
    ax_eta.set_xlabel(r"$\tau_{\rm T}$")
    ax_eta.set_ylabel(r"$\eta=L_{\rm C,down}/(L_{\rm C,up}+L_{\rm C,down})$")
    ax_eta.set_title("compPSc eta depends on both kTe and tau")
    ax_eta.grid(True, alpha=0.25, linestyle=":")
    ax_eta.legend(title=r"$kT_{\rm e}$", ncols=2, frameon=False, fontsize=8.5)

    tau = np.array([float(row["tau_T"]) for row in summary], dtype=float)
    rel_range = np.array([float(row["eta_relative_range_over_kTe"]) for row in summary], dtype=float)
    ax_range.plot(tau, rel_range, color="#CC79A7", marker="o", lw=2.0)
    ax_range.axhline(0.08, color="#555555", lw=1.0, ls="--")
    ax_range.set_xscale("log")
    ax_range.set_yscale("log")
    ax_range.set_xlabel(r"$\tau_{\rm T}$")
    ax_range.set_ylabel(r"relative range of $\eta$ over $kT_{\rm e}$")
    ax_range.set_title("One-dimensional eta(tau) is not adequate")
    ax_range.grid(True, alpha=0.25, linestyle=":")

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ETA_PNG, dpi=200)
    plt.close(fig)


def plot_albedo_comparison(
    rows: Sequence[dict[str, float | int | str | bool]],
    *,
    kTe_values: Sequence[float],
    tau_values: Sequence[float],
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_comppsc_eta_grid")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    datasets = [
        ("effective_albedo_reflect", "reflect"),
        ("effective_albedo_ireflect_n1e13", r"ireflect, $n=10^{13}$ cm$^{-3}$"),
        ("effective_albedo_ireflect_n1e15", r"ireflect, $n=10^{15}$ cm$^{-3}$"),
    ]
    matrices = [rows_to_matrix(rows, value_key=key, kTe_values=kTe_values, tau_values=tau_values) for key, _ in datasets]
    positive = np.concatenate([matrix[np.isfinite(matrix) & (matrix > 0.0)] for matrix in matrices])
    vmin = max(float(np.nanmin(positive)), 1.0e-8)
    vmax = float(np.nanmax(positive))
    x_edges = geometric_edges(kTe_values)
    y_edges = geometric_edges(tau_values)
    valid = rows_to_matrix(rows, value_key="converged", kTe_values=kTe_values, tau_values=tau_values)

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.0), sharex=True, sharey=True, constrained_layout=True)
    mesh = None
    for ax, matrix, (_, title) in zip(axes, matrices, datasets):
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            matrix,
            shading="auto",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap="magma",
        )
        bad_tau, bad_kte = np.where(valid < 0.5)
        if bad_tau.size:
            ax.scatter(
                np.asarray(kTe_values)[bad_kte],
                np.asarray(tau_values)[bad_tau],
                marker="x",
                s=28,
                color="white",
                linewidths=1.2,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$kT_{\rm e}$ (keV)")
        ax.set_title(title)
        ax.grid(True, alpha=0.18, linestyle=":")
    axes[0].set_ylabel(r"$\tau_{\rm T}$")
    assert mesh is not None
    colorbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), pad=0.018, shrink=0.9)
    colorbar.set_label("effective albedo")
    fig.suptitle(r"Effective disk albedo from the same downward compPSc illumination")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ALBEDO_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_fixed_xi_albedo(
    rows: Sequence[dict[str, float | int | str | bool]],
    *,
    kTe_values: Sequence[float],
    tau_values: Sequence[float],
    fixed_xi: float,
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_comppsc_eta_grid")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    _, _, _, output_png = fixed_xi_output_paths(fixed_xi)
    value_key = fixed_xi_column(fixed_xi)
    matrix = rows_to_matrix(rows, value_key=value_key, kTe_values=kTe_values, tau_values=tau_values)
    valid = rows_to_matrix(rows, value_key="converged", kTe_values=kTe_values, tau_values=tau_values)
    positive = matrix[np.isfinite(matrix) & (matrix > 0.0)]
    x_edges = geometric_edges(kTe_values)
    y_edges = geometric_edges(tau_values)

    fig, ax = plt.subplots(figsize=(7.1, 5.2), constrained_layout=True)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        shading="auto",
        norm=LogNorm(vmin=max(float(np.nanmin(positive)), 1.0e-8), vmax=float(np.nanmax(positive))),
        cmap="magma",
    )
    bad_tau, bad_kte = np.where(valid < 0.5)
    if bad_tau.size:
        ax.scatter(
            np.asarray(kTe_values)[bad_kte],
            np.asarray(tau_values)[bad_tau],
            marker="x",
            s=34,
            color="white",
            linewidths=1.3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$kT_{\rm e}$ (keV)")
    ax.set_ylabel(r"$\tau_{\rm T}$")
    ax.set_title(rf"ireflect effective albedo, fixed $\xi={fixed_xi:g}$")
    ax.grid(True, alpha=0.18, linestyle=":")
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label("effective albedo")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_float_list(values: list[str] | None, defaults: Sequence[float]) -> list[float]:
    if not values:
        return list(defaults)
    parsed: list[float] = []
    for value in values:
        parsed.extend(float(item) for item in value.split(",") if item)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kTe-grid", nargs="*", default=None, help="Space/comma separated kTe values in keV.")
    parser.add_argument("--tau-grid", nargs="*", default=None, help="Space/comma separated tau_T values.")
    parser.add_argument("--densities", nargs="*", default=None, help="Space/comma separated reflector densities in cm^-3.")
    parser.add_argument("--max-scatter", type=int, default=2000)
    parser.add_argument("--convergence-tolerance", type=float, default=3.2e-3)
    parser.add_argument("--tbb-kev", type=float, default=0.005)
    parser.add_argument("--mu-order", type=int, default=8)
    parser.add_argument("--observer-mu", type=float, default=0.5)
    parser.add_argument("--no-exact-angles", action="store_true")
    parser.add_argument("--fixed-xi-only", action="store_true", help="Only compute a fixed-xi ireflect albedo table.")
    parser.add_argument("--fixed-xi", type=float, default=100.0, help="Ionization parameter used with --fixed-xi-only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kTe_values = parse_float_list(args.kTe_grid, DEFAULT_KTE_GRID)
    tau_values = parse_float_list(args.tau_grid, DEFAULT_TAU_GRID)
    densities_cm3 = parse_float_list(args.densities, DEFAULT_DENSITIES)
    if densities_cm3 != DEFAULT_DENSITIES:
        raise ValueError("This scanner currently writes fixed n=1e13/n=1e15 albedo columns.")

    if args.fixed_xi_only:
        rows = scan_fixed_xi_grid(
            kTe_values=kTe_values,
            tau_values=tau_values,
            fixed_xi=args.fixed_xi,
            max_scatter=args.max_scatter,
            convergence_tolerance=args.convergence_tolerance,
            tbb_kev=args.tbb_kev,
            hemisphere_mu_order=args.mu_order,
            observer_mu=args.observer_mu,
            exact_angles=not args.no_exact_angles,
        )
        long_csv, matrix_csv, valid_csv, output_png = fixed_xi_output_paths(args.fixed_xi)
        value_key = fixed_xi_column(args.fixed_xi)
        albedo_matrix = rows_to_matrix(rows, value_key=value_key, kTe_values=kTe_values, tau_values=tau_values)
        valid_matrix = rows_to_matrix(rows, value_key="converged", kTe_values=kTe_values, tau_values=tau_values)
        write_matrix_csv(matrix_csv, albedo_matrix, kTe_values=kTe_values, tau_values=tau_values, value_prefix=value_key)
        write_matrix_csv(valid_csv, valid_matrix, kTe_values=kTe_values, tau_values=tau_values, value_prefix="valid")
        plot_fixed_xi_albedo(rows, kTe_values=kTe_values, tau_values=tau_values, fixed_xi=args.fixed_xi)
        print(long_csv)
        print(matrix_csv)
        print(valid_csv)
        print(output_png)
        return

    rows = scan_grid(
        kTe_values=kTe_values,
        tau_values=tau_values,
        densities_cm3=densities_cm3,
        max_scatter=args.max_scatter,
        convergence_tolerance=args.convergence_tolerance,
        tbb_kev=args.tbb_kev,
        hemisphere_mu_order=args.mu_order,
        observer_mu=args.observer_mu,
        exact_angles=not args.no_exact_angles,
    )

    eta_matrix = rows_to_matrix(rows, value_key="eta", kTe_values=kTe_values, tau_values=tau_values)
    valid_matrix = rows_to_matrix(rows, value_key="converged", kTe_values=kTe_values, tau_values=tau_values)
    a_model_matrix = rows_to_converged_matrix(rows, value_key="A_model", kTe_values=kTe_values, tau_values=tau_values)
    p_sc_matrix = rows_to_converged_matrix(rows, value_key="p_sc", kTe_values=kTe_values, tau_values=tau_values)
    eta_summary = summarize_eta_dependence(rows, tau_values)
    albedo_summary = summarize_albedo(rows)

    write_matrix_csv(ETA_MATRIX_CSV, eta_matrix, kTe_values=kTe_values, tau_values=tau_values, value_prefix="eta")
    write_matrix_csv(ETA_VALID_MASK_CSV, valid_matrix, kTe_values=kTe_values, tau_values=tau_values, value_prefix="valid")
    write_matrix_csv(A_MODEL_MATRIX_CSV, a_model_matrix, kTe_values=kTe_values, tau_values=tau_values, value_prefix="A_model")
    write_matrix_csv(P_SC_MATRIX_CSV, p_sc_matrix, kTe_values=kTe_values, tau_values=tau_values, value_prefix="p_sc")
    write_rows(ETA_SUMMARY_CSV, eta_summary)
    write_rows(ALBEDO_SUMMARY_CSV, albedo_summary)
    plot_eta_dependence(rows, kTe_values=kTe_values, tau_values=tau_values)
    plot_albedo_comparison(rows, kTe_values=kTe_values, tau_values=tau_values)

    print(LONG_CSV)
    print(ETA_MATRIX_CSV)
    print(ETA_VALID_MASK_CSV)
    print(A_MODEL_MATRIX_CSV)
    print(P_SC_MATRIX_CSV)
    print(ETA_SUMMARY_CSV)
    print(ALBEDO_SUMMARY_CSV)
    print(ETA_PNG)
    print(ALBEDO_PNG)


if __name__ == "__main__":
    main()
