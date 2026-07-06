#!/usr/bin/env python3
"""Compare local compPSc and XSPEC compTT above 10 keV."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


KTE_VALUES = (10.0, 20.0, 51.1, 100.0, 255.5)
TAU_VALUES = (0.1, 0.3, 1.0, 2.0, 3.0, 5.0)
TAU_FACTORS = (0.5, 1.0, 2.0)

GLOBAL_CONFIG = {
    "seed_kT_keV": 0.005,
    "compps_geometry": 1.0,
    "compps_cos_incl": 0.5,
    "comptt_approx": 1.0,
    "normalization_energy_keV": 10.0,
    "energy_min_keV": 1e-3,
    "energy_max_keV": 1e3,
    "energy_bins": 1200,
    "flux_floor": 1e-8,
    "median_limit": 0.08,
    "p95_limit": 0.08,
    "high_tau_scatter_scale": 30.0,
}


def find_normalization_bin(edges, energy_keV=10.0):
    """Return the half-open bin index containing the normalization energy."""
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("edges must be a one-dimensional array with at least two values")
    if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
        raise ValueError("edges must be finite and strictly increasing")
    if not edges[0] <= energy_keV < edges[-1]:
        raise ValueError("normalization energy lies outside the energy grid")
    return int(np.searchsorted(edges, energy_keV, side="right") - 1)


def normalize_and_compare(edges, comp_ps, comp_tt, energy_keV=10.0, flux_floor=1e-8):
    """Normalize in one bin and compare compTT to compPSc at higher energies."""
    edges = np.asarray(edges, dtype=float)
    comp_ps = np.asarray(comp_ps, dtype=float)
    comp_tt = np.asarray(comp_tt, dtype=float)
    if comp_ps.shape != comp_tt.shape or comp_ps.shape != (edges.size - 1,):
        raise ValueError("spectra must have one value per energy bin")
    if not 0.0 <= flux_floor < 1.0:
        raise ValueError("flux_floor must be in [0, 1)")

    normalization_bin = find_normalization_bin(edges, energy_keV)
    ps_norm_value = comp_ps[normalization_bin]
    tt_norm_value = comp_tt[normalization_bin]
    if not np.isfinite(ps_norm_value) or ps_norm_value <= 0.0:
        raise ValueError("compPSc normalization-bin flux must be finite and positive")
    if not np.isfinite(tt_norm_value) or tt_norm_value <= 0.0:
        raise ValueError("compTT normalization-bin flux must be finite and positive")

    comp_ps_normalized = comp_ps / ps_norm_value
    comp_tt_normalized = comp_tt / tt_norm_value
    in_high_energy_band = np.arange(comp_ps.size) >= normalization_bin
    finite_positive = (
        in_high_energy_band
        & np.isfinite(comp_ps_normalized)
        & np.isfinite(comp_tt_normalized)
        & (comp_ps_normalized > 0.0)
        & (comp_tt_normalized > 0.0)
    )
    if not np.any(finite_positive):
        raise ValueError("no finite positive high-energy bins are available")

    ps_peak = np.max(comp_ps_normalized[finite_positive])
    tt_peak = np.max(comp_tt_normalized[finite_positive])
    valid = (
        finite_positive
        & (comp_ps_normalized >= flux_floor * ps_peak)
        & (comp_tt_normalized >= flux_floor * tt_peak)
    )
    if not np.any(valid):
        raise ValueError("no high-energy bins remain above the flux floor")

    residual = np.full(comp_ps.shape, np.nan, dtype=float)
    residual[valid] = comp_tt_normalized[valid] / comp_ps_normalized[valid] - 1.0
    abs_residual = np.abs(residual[valid])
    return {
        "normalization_bin": normalization_bin,
        "comp_ps_norm_value": float(ps_norm_value),
        "comp_tt_norm_value": float(tt_norm_value),
        "comp_ps_normalized": comp_ps_normalized,
        "comp_tt_normalized": comp_tt_normalized,
        "valid": valid,
        "residual": residual,
        "n_valid": int(np.count_nonzero(valid)),
        "median_abs_rel": float(np.median(abs_residual)),
        "p95_abs_rel": float(np.percentile(abs_residual, 95.0)),
        "max_abs_rel": float(np.max(abs_residual)),
    }


def passes_acceptance(metrics, median_limit=0.08, p95_limit=0.08):
    """Return whether both metrics are within the inclusive 8% limits."""
    median = float(metrics["median_abs_rel"])
    p95 = float(metrics["p95_abs_rel"])
    return (
        np.isfinite(median)
        and np.isfinite(p95)
        and median <= median_limit
        and p95 <= p95_limit
    )


def default_compps_max_scatter(tau):
    """Return the native compPS scattering-order truncation for a depth."""
    return 50 + int(4.0 * float(tau) ** 2)


def select_compps_max_scatter(tau):
    """Use a larger scattering-order truncation for optically thick slabs."""
    tau = float(tau)
    default = default_compps_max_scatter(tau)
    if tau < 2.0:
        return default
    high_tau = 50 + int(GLOBAL_CONFIG["high_tau_scatter_scale"] * tau**2)
    return max(default, high_tau)


def snapshot_spectrum(values):
    """Copy XSPEC model values before later model mutations or clearing."""
    return np.array(values, dtype=float, copy=True)


def make_temperature_figure_title(kTe):
    """Return the two-line title used by each temperature comparison figure."""
    return (
        rf"compPSc vs compTT above 10 keV, $kT_e={kTe:g}$ keV"
        "\nsolid: compPSc; dashed: compTT"
    )


def heatmap_color_limits(values):
    """Return finite positive limits that retain the full metric range."""
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        raise ValueError("heatmap values must contain at least one finite positive value")
    return float(np.min(positive)), float(np.max(positive))


def comparison_figure_layout(tau_factors):
    """Return figure rows and columns for the configured tau mappings."""
    tau_factors = tuple(tau_factors)
    if not tau_factors:
        raise ValueError("at least one tau factor is required")
    return 2, len(tau_factors)


def format_tau_factor(factor):
    """Format tau mapping factors without losing non-integer values."""
    return f"{float(factor):g}"


def heatmap_annotation_color(normed_value):
    """Choose readable annotation text for dark and bright heatmap colors."""
    value = float(normed_value)
    return "white" if value < 0.25 or value > 0.72 else "black"


def _set_compps_parameters(model, kTe, tau):
    max_scatter = select_compps_max_scatter(tau)
    model.setPars(
        {
            1: kTe,
            2: 2.0,
            3: -1.0,
            4: 1000.0,
            6: GLOBAL_CONFIG["compps_geometry"],
            7: 1.0,
            8: GLOBAL_CONFIG["compps_cos_incl"],
            9: 1.0,
            10: 0.0,
            11: 1.0,
            12: float(max_scatter),
            13: GLOBAL_CONFIG["seed_kT_keV"],
            14: 1.0,
        }
    )
    model(5).values = [tau, 0.1, 0.001, 0.001, 10.0, 10.0]
    return max_scatter


def _set_comptt_parameters(model, kTe, tau):
    model.setPars(
        0.0,
        GLOBAL_CONFIG["seed_kT_keV"],
        kTe,
        tau,
        GLOBAL_CONFIG["comptt_approx"],
        1.0,
    )


def evaluate_xspec_models(model_dir):
    """Evaluate all compPSc and compTT cases on a common energy grid."""
    try:
        from xspec import AllModels, Model, Xset
    except ImportError as exc:
        raise RuntimeError("run this script inside the heasoft_full environment") from exc

    Xset.chatter = 5
    AllModels.clear()
    AllModels.lmod("comppsc", str(model_dir))
    AllModels.setEnergies(
        f"{GLOBAL_CONFIG['energy_min_keV']} "
        f"{GLOBAL_CONFIG['energy_max_keV']} "
        f"{GLOBAL_CONFIG['energy_bins']} log"
    )

    comp_ps_spectra = {}
    comp_ps_max_scatter = {}
    comp_ps_model = Model("comppsc*bbodyrad")
    for kTe in KTE_VALUES:
        for tau in TAU_VALUES:
            comp_ps_max_scatter[(kTe, tau)] = _set_compps_parameters(
                comp_ps_model, kTe, tau
            )
            edges = np.asarray(comp_ps_model.energies(0), dtype=float)
            comp_ps_spectra[(kTe, tau)] = snapshot_spectrum(comp_ps_model.values(0))

    AllModels.clear()
    comp_tt_spectra = {}
    comp_tt_model = Model("compTT")
    for kTe in KTE_VALUES:
        for tau in TAU_VALUES:
            for factor in TAU_FACTORS:
                _set_comptt_parameters(comp_tt_model, kTe, factor * tau)
                comp_tt_spectra[(kTe, tau, factor)] = snapshot_spectrum(
                    comp_tt_model.values(0)
                )

    AllModels.clear()
    return edges, comp_ps_spectra, comp_ps_max_scatter, comp_tt_spectra


def build_comparisons(edges, comp_ps_spectra, comp_ps_max_scatter, comp_tt_spectra):
    """Normalize and compare every temperature, depth, and mapping case."""
    comparisons = {}
    for kTe in KTE_VALUES:
        for tau in TAU_VALUES:
            for factor in TAU_FACTORS:
                metrics = normalize_and_compare(
                    edges,
                    comp_ps_spectra[(kTe, tau)],
                    comp_tt_spectra[(kTe, tau, factor)],
                    energy_keV=GLOBAL_CONFIG["normalization_energy_keV"],
                    flux_floor=GLOBAL_CONFIG["flux_floor"],
                )
                metrics["pass"] = passes_acceptance(
                    metrics,
                    median_limit=GLOBAL_CONFIG["median_limit"],
                    p95_limit=GLOBAL_CONFIG["p95_limit"],
                )
                metrics["comp_ps_max_scatter"] = comp_ps_max_scatter[(kTe, tau)]
                comparisons[(kTe, tau, factor)] = metrics
    return comparisons


def write_csv_outputs(output_dir, edges, comparisons):
    """Write case-level and bin-level validation tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    summary_rows = []
    spectra_rows = []

    for kTe in KTE_VALUES:
        for tau in TAU_VALUES:
            for factor in TAU_FACTORS:
                result = comparisons[(kTe, tau, factor)]
                norm_bin = result["normalization_bin"]
                summary_rows.append(
                    {
                        "kTe_keV": kTe,
                        "tau_compPS": tau,
                        "tau_factor": factor,
                        "tau_compTT": factor * tau,
                        "compPSc_max_scatter": result["comp_ps_max_scatter"],
                        "normalization_bin": norm_bin,
                        "normalization_E_low_keV": edges[norm_bin],
                        "normalization_E_high_keV": edges[norm_bin + 1],
                        "normalization_E_center_keV": centers[norm_bin],
                        "compPSc_raw_norm_bin": result["comp_ps_norm_value"],
                        "compTT_raw_norm_bin": result["comp_tt_norm_value"],
                        "n_valid": result["n_valid"],
                        "median_abs_rel": result["median_abs_rel"],
                        "p95_abs_rel": result["p95_abs_rel"],
                        "max_abs_rel": result["max_abs_rel"],
                        "pass": result["pass"],
                    }
                )
                for index, center in enumerate(centers):
                    spectra_rows.append(
                        {
                            "kTe_keV": kTe,
                            "tau_compPS": tau,
                            "tau_factor": factor,
                            "tau_compTT": factor * tau,
                            "E_low_keV": edges[index],
                            "E_high_keV": edges[index + 1],
                            "E_center_keV": center,
                            "compPSc_normalized": result["comp_ps_normalized"][index],
                            "compTT_normalized": result["comp_tt_normalized"][index],
                            "valid": bool(result["valid"][index]),
                            "residual": result["residual"][index],
                        }
                    )

    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "spectra.csv", spectra_rows)
    return summary_rows


def write_figures(output_dir, edges, comparisons):
    """Write temperature-specific comparisons and metric heatmaps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    centers = np.sqrt(edges[:-1] * edges[1:])
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(TAU_VALUES)))
    figure_rows, figure_columns = comparison_figure_layout(TAU_FACTORS)

    for kTe in KTE_VALUES:
        fig, axes = plt.subplots(
            figure_rows, figure_columns, figsize=(5.8 * figure_columns, 8), sharex=True, squeeze=False
        )
        for column, factor in enumerate(TAU_FACTORS):
            ax_spectrum = axes[0, column]
            ax_residual = axes[1, column]
            for color, tau in zip(colors, TAU_VALUES):
                result = comparisons[(kTe, tau, factor)]
                valid = result["valid"]
                label = rf"$\tau_{{PS}}={tau:g}$"
                ax_spectrum.loglog(
                    centers[valid],
                    result["comp_ps_normalized"][valid],
                    color=color,
                    linewidth=1.4,
                    label=label,
                )
                ax_spectrum.loglog(
                    centers[valid],
                    result["comp_tt_normalized"][valid],
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                )
                ax_residual.semilogx(
                    centers[valid], 100.0 * result["residual"][valid], color=color, linewidth=1.2
                )

            ax_spectrum.axvline(10.0, color="#555555", linewidth=0.8, linestyle=":")
            ax_spectrum.set_title(rf"$\tau_{{TT}}={factor:g}\,\tau_{{PS}}$")
            ax_spectrum.set_ylabel("normalized photons/bin")
            ax_spectrum.grid(alpha=0.2, which="both")
            ax_residual.axhline(0.0, color="#202124", linewidth=0.8)
            ax_residual.axhline(8.0, color="#9c2f2f", linewidth=0.8, linestyle=":")
            ax_residual.axhline(-8.0, color="#9c2f2f", linewidth=0.8, linestyle=":")
            ax_residual.set_xlabel("Energy (keV)")
            ax_residual.set_ylabel("compTT / compPSc - 1 (%)")
            ax_residual.grid(alpha=0.2, which="both")

        axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
        fig.suptitle(make_temperature_figure_title(kTe), fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        tag = str(kTe).replace(".", "p")
        fig.savefig(output_dir / f"spectra_residuals_kTe_{tag}_keV.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(
        figure_rows, figure_columns, figsize=(6.5 * figure_columns, 7), constrained_layout=True, squeeze=False
    )
    all_percent = [
        100.0 * comparisons[key][metric]
        for key in comparisons
        for metric in ("median_abs_rel", "p95_abs_rel")
    ]
    vmin, vmax = heatmap_color_limits(all_percent)
    color_norm = LogNorm(vmin=vmin, vmax=vmax)
    for column, factor in enumerate(TAU_FACTORS):
        for row, (metric, title) in enumerate(
            (("median_abs_rel", "Median error"), ("p95_abs_rel", "P95 error"))
        ):
            matrix = np.array(
                [
                    [100.0 * comparisons[(kTe, tau, factor)][metric] for tau in TAU_VALUES]
                    for kTe in KTE_VALUES
                ]
            )
            ax = axes[row, column]
            image = ax.imshow(
                matrix, aspect="auto", origin="lower", norm=color_norm, cmap="magma"
            )
            for iy in range(matrix.shape[0]):
                for ix in range(matrix.shape[1]):
                    color = heatmap_annotation_color(image.norm(matrix[iy, ix]))
                    ax.text(ix, iy, f"{matrix[iy, ix]:.1f}", ha="center", va="center", color=color, fontsize=8)
            ax.set_xticks(range(len(TAU_VALUES)), [f"{tau:g}" for tau in TAU_VALUES])
            ax.set_yticks(range(len(KTE_VALUES)), [f"{kTe:g}" for kTe in KTE_VALUES])
            ax.set_xlabel(r"$\tau_{compPS}$")
            ax.set_ylabel(r"$kT_e$ (keV)")
            ax.set_title(rf"{title} (%), $\tau_{{TT}}={factor:g}\tau_{{PS}}$")
            fig.colorbar(image, ax=ax, label="error (%)")
    fig.savefig(output_dir / "metric_heatmaps.png", dpi=180)
    plt.close(fig)


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    model_dir = Path(__file__).resolve().parent
    output_dir = model_dir / "validation_comppsc_vs_comptt_high_energy"
    edges, comp_ps_spectra, comp_ps_max_scatter, comp_tt_spectra = evaluate_xspec_models(
        model_dir
    )
    comparisons = build_comparisons(
        edges, comp_ps_spectra, comp_ps_max_scatter, comp_tt_spectra
    )
    summary = write_csv_outputs(output_dir, edges, comparisons)
    write_figures(output_dir, edges, comparisons)

    print("kTe tau_PS maxSc factor tau_TT median p95 max pass")
    for row in summary:
        print(
            f"{row['kTe_keV']:6.1f} {row['tau_compPS']:5.1f} "
            f"{row['compPSc_max_scatter']:5d} {format_tau_factor(row['tau_factor']):>5} "
            f"{row['tau_compTT']:6.1f} {row['median_abs_rel']:.6g} "
            f"{row['p95_abs_rel']:.6g} {row['max_abs_rel']:.6g} {row['pass']}"
        )
    print(f"Wrote validation artifacts to {output_dir}")
    return 0 if all(row["pass"] for row in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
