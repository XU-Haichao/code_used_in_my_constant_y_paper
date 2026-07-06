#!/usr/bin/env python3
"""Compare local compPSc against XSPEC compps for a 5 eV slab seed."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


VALIDATION_CASES = (
    (51.1, 0.1),
    (51.1, 1.0),
    (255.5, 0.1),
    (255.5, 1.0),
)

GLOBAL_CONFIG = {
    "seed_kT_keV": 0.005,
    "geometry": 1.0,
    "cos_incl": 0.5,
    "reflection": 0.0,
    "energy_min_keV": 1e-3,
    "energy_max_keV": 1e3,
    "energy_bins": 1200,
    "comparison_min_keV": 2e-3,
}


def compare_spectra(reference, candidate, flux_floor=1e-8):
    """Compare spectral shapes after removing one multiplicative scale."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have identical shapes")
    if not 0.0 <= flux_floor < 1.0:
        raise ValueError("flux_floor must be in [0, 1)")

    finite_positive = (
        np.isfinite(reference)
        & np.isfinite(candidate)
        & (reference > 0.0)
        & (candidate > 0.0)
    )
    if not np.any(finite_positive):
        raise ValueError("no finite positive bins are available")

    reference_peak = np.max(reference[finite_positive])
    candidate_peak = np.max(candidate[finite_positive])
    valid = (
        finite_positive
        & (reference >= flux_floor * reference_peak)
        & (candidate >= flux_floor * candidate_peak)
    )
    if not np.any(valid):
        raise ValueError("no bins remain above the flux floor")

    scale = float(np.exp(np.median(np.log(reference[valid] / candidate[valid]))))
    residual = np.full(reference.shape, np.nan, dtype=float)
    residual[valid] = scale * candidate[valid] / reference[valid] - 1.0
    abs_residual = np.abs(residual[valid])

    return {
        "scale": scale,
        "valid": valid,
        "residual": residual,
        "n_valid": int(np.count_nonzero(valid)),
        "median_abs_rel": float(np.median(abs_residual)),
        "p95_abs_rel": float(np.percentile(abs_residual, 95.0)),
        "max_abs_rel": float(np.max(abs_residual)),
    }


def passes_acceptance(metrics, median_limit=0.01, p95_limit=0.03):
    """Return whether shape metrics satisfy the agreed strict limits."""
    median = float(metrics["median_abs_rel"])
    p95 = float(metrics["p95_abs_rel"])
    return (
        np.isfinite(median)
        and np.isfinite(p95)
        and median < median_limit
        and p95 < p95_limit
    )


def _set_builtin_parameters(model, kTe, tau):
    values = (
        kTe,
        2.0,
        -1.0,
        1000.0,
        GLOBAL_CONFIG["seed_kT_keV"],
        tau,
        GLOBAL_CONFIG["geometry"],
        1.0,
        GLOBAL_CONFIG["cos_incl"],
        1.0,
        GLOBAL_CONFIG["reflection"],
        1.0,
        1.0,
        0.0,
        1e6,
        -10.0,
        10.0,
        1000.0,
        0.0,
        1.0,
    )
    model.setPars(*values)


def _set_custom_parameters(model, kTe, tau):
    values = (
        kTe,
        2.0,
        -1.0,
        1000.0,
        tau,
        GLOBAL_CONFIG["geometry"],
        1.0,
        GLOBAL_CONFIG["cos_incl"],
        1.0,
        0.0,
        1.0,
        GLOBAL_CONFIG["seed_kT_keV"],
        1.0,
    )
    model.setPars(*values)


def evaluate_xspec_models(model_dir):
    """Evaluate built-in and local models on the same PyXspec grid."""
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

    reference = {}
    builtin = Model("compps")
    for kTe, tau in VALIDATION_CASES:
        _set_builtin_parameters(builtin, kTe, tau)
        edges = np.asarray(builtin.energies(0), dtype=float)
        reference[(kTe, tau)] = np.asarray(builtin.values(0), dtype=float)

    AllModels.clear()
    custom = Model("comppsc*bbodyrad")
    candidate = {}
    for kTe, tau in VALIDATION_CASES:
        _set_custom_parameters(custom, kTe, tau)
        candidate[(kTe, tau)] = np.asarray(custom.values(0), dtype=float)

    AllModels.clear()
    return edges, reference, candidate


def write_outputs(output_dir, edges, reference, candidate):
    """Write per-bin values, summary statistics, and a comparison figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = np.diff(edges)
    spectra_rows = []
    summary_rows = []
    comparisons = {}

    for kTe, tau in VALIDATION_CASES:
        ref = reference[(kTe, tau)]
        cand = candidate[(kTe, tau)]
        in_comparison_band = centers >= GLOBAL_CONFIG["comparison_min_keV"]
        metrics = compare_spectra(
            np.where(in_comparison_band, ref, np.nan),
            np.where(in_comparison_band, cand, np.nan),
        )
        passed = passes_acceptance(metrics)
        comparisons[(kTe, tau)] = metrics
        summary_rows.append(
            {
                "kTe_keV": kTe,
                "tau": tau,
                "seed_kT_keV": GLOBAL_CONFIG["seed_kT_keV"],
                "geom": GLOBAL_CONFIG["geometry"],
                "cosIncl": GLOBAL_CONFIG["cos_incl"],
                "comparison_min_keV": GLOBAL_CONFIG["comparison_min_keV"],
                "scale_candidate_to_reference": metrics["scale"],
                "candidate_over_reference_norm": 1.0 / metrics["scale"],
                "n_valid": metrics["n_valid"],
                "median_abs_rel": metrics["median_abs_rel"],
                "p95_abs_rel": metrics["p95_abs_rel"],
                "max_abs_rel": metrics["max_abs_rel"],
                "pass": passed,
            }
        )
        for index, energy in enumerate(centers):
            spectra_rows.append(
                {
                    "kTe_keV": kTe,
                    "tau": tau,
                    "E_low_keV": edges[index],
                    "E_high_keV": edges[index + 1],
                    "E_center_keV": energy,
                    "builtin_compps_photons_per_bin": ref[index],
                    "custom_comppsc_photons_per_bin": cand[index],
                    "scaled_custom_photons_per_bin": metrics["scale"] * cand[index],
                    "valid": bool(metrics["valid"][index]),
                    "shape_residual": metrics["residual"][index],
                }
            )

    _write_csv(output_dir / "spectra.csv", spectra_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)

    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharex="col")
    for row, (kTe, tau) in enumerate(VALIDATION_CASES):
        ref = reference[(kTe, tau)]
        cand = candidate[(kTe, tau)]
        metrics = comparisons[(kTe, tau)]
        valid = metrics["valid"]
        spectral_weight = centers**2 / widths

        ax_spectrum, ax_residual = axes[row]
        ax_spectrum.loglog(
            centers[valid],
            spectral_weight[valid] * ref[valid],
            color="#202124",
            linewidth=1.8,
            label="XSPEC compps",
        )
        ax_spectrum.loglog(
            centers[valid],
            spectral_weight[valid] * metrics["scale"] * cand[valid],
            color="#c23b22",
            linewidth=1.2,
            linestyle="--",
            label="scaled compPSc*bbodyrad",
        )
        ax_spectrum.set_ylabel(r"$E^2\,dN/dE$ (arb.)")
        ax_spectrum.set_title(rf"$kT_e={kTe:g}$ keV, $\tau={tau:g}$")
        ax_spectrum.grid(alpha=0.2, which="both")
        if row == 0:
            ax_spectrum.legend(frameon=False, fontsize=9)

        ax_residual.semilogx(
            centers[valid], 100.0 * metrics["residual"][valid], color="#2457a6", linewidth=1.0
        )
        ax_residual.axhline(0.0, color="#202124", linewidth=0.8)
        ax_residual.axhline(3.0, color="#777777", linewidth=0.7, linestyle=":")
        ax_residual.axhline(-3.0, color="#777777", linewidth=0.7, linestyle=":")
        ax_residual.set_ylabel("shape residual (%)")
        ax_residual.set_title(
            f"median={100 * metrics['median_abs_rel']:.2f}%, "
            f"p95={100 * metrics['p95_abs_rel']:.2f}%"
        )
        ax_residual.grid(alpha=0.2, which="both")

    axes[-1, 0].set_xlabel("Energy (keV)")
    axes[-1, 1].set_xlabel("Energy (keV)")
    fig.suptitle("5 eV blackbody seed, slab geometry, cosIncl=0.5", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_dir / "compps_bb5ev_slab_comparison.png", dpi=180)
    plt.close(fig)
    return summary_rows


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    model_dir = Path(__file__).resolve().parent
    output_dir = model_dir / "validation_bb5ev_slab"
    edges, reference, candidate = evaluate_xspec_models(model_dir)
    summary = write_outputs(output_dir, edges, reference, candidate)

    print("kTe_keV tau scale median_abs_rel p95_abs_rel max_abs_rel pass")
    for row in summary:
        print(
            f"{row['kTe_keV']:7.1f} {row['tau']:3.1f} "
            f"{row['scale_candidate_to_reference']:.8g} "
            f"{row['median_abs_rel']:.6g} {row['p95_abs_rel']:.6g} "
            f"{row['max_abs_rel']:.6g} {row['pass']}"
        )
    print(f"Wrote validation artifacts to {output_dir}")
    return 0 if all(row["pass"] for row in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
