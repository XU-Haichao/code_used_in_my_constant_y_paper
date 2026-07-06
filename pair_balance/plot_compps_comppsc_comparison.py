#!/usr/bin/env python3
"""Compare compPS and compPSc slab pair-balance curves."""

from __future__ import annotations

import csv
import os
import pathlib

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_compps_comppsc_compare")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
PAIR_DATA = ROOT / "pair_balance" / "data"
OUTPUT = ROOT / "output"

COMPPS_CSV = PAIR_DATA / "ps96_slab_dratio_tau_kTe_theta_0.02_1.0_log40.csv"
COMPPSC_CSV = PAIR_DATA / "comppsc_slab_dratio_tau_kTe_theta_0.02_1_log40_maxsc2000.csv"
COMPARISON_PNG = OUTPUT / "compps_comppsc_pair_balance_comparison.png"
RATIO_PNG = OUTPUT / "compps_comppsc_pair_balance_ratios.png"
MATCHED_CSV = OUTPUT / "compps_comppsc_pair_balance_matched.csv"

COLORS = {
    0.0: "#0072B2",
    0.25: "#009E73",
    1.0: "#D55E00",
    3.0: "#8B5CF6",
}


def load_rows(path: pathlib.Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            parsed: dict[str, float | str] = {}
            for key, value in raw.items():
                if value is None or value == "":
                    raise ValueError(f"Missing {key!r} in {path}")
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def row_key(row: dict[str, float | str]) -> tuple[float, float]:
    return float(row["d_ratio"]), round(float(row["theta"]), 12)


def match_rows(
    compps_rows: list[dict[str, float | str]],
    comppsc_rows: list[dict[str, float | str]],
) -> list[dict[str, float]]:
    old = {row_key(row): row for row in compps_rows}
    new = {row_key(row): row for row in comppsc_rows}
    if old.keys() != new.keys():
        missing_new = sorted(old.keys() - new.keys())
        missing_old = sorted(new.keys() - old.keys())
        raise ValueError(
            f"Scan grids differ: missing in compPSc={missing_new}; "
            f"missing in compPS={missing_old}"
        )

    matched: list[dict[str, float]] = []
    for key in sorted(old):
        old_row = old[key]
        new_row = new[key]
        tau_old = float(old_row["tau_T"])
        tau_new = float(new_row["tau_T"])
        ldiss_old = float(old_row["l_diss_local"])
        ldiss_new = float(new_row["l_diss_local"])
        matched.append(
            {
                "d_ratio": key[0],
                "theta": float(old_row["theta"]),
                "kTe_keV": float(old_row["kTe_keV"]),
                "tau_T_compps": tau_old,
                "tau_T_comppsc": tau_new,
                "tau_fractional_difference": tau_new / tau_old - 1.0,
                "l_diss_compps": ldiss_old,
                "l_diss_comppsc": ldiss_new,
                "l_diss_ratio": ldiss_new / ldiss_old,
            }
        )
    return matched


def configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "axes.facecolor": "#fbfbf8",
            "figure.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 10.5,
            "legend.frameon": False,
        }
    )


def curve(rows: list[dict[str, float]], d_ratio: float) -> list[dict[str, float]]:
    return sorted(
        (row for row in rows if row["d_ratio"] == d_ratio),
        key=lambda row: row["theta"],
    )


def plot_comparison(rows: list[dict[str, float]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2))
    panels = (
        (
            axes[0],
            "kTe_keV",
            ("tau_T_compps", "tau_T_comppsc"),
            r"$kT_{\rm e}$ (keV)",
            r"$\tau_{\rm T}$",
            r"Energy balance: $\tau_{\rm T}$-$kT_{\rm e}$",
        ),
        (
            axes[1],
            "kTe_keV",
            ("l_diss_compps", "l_diss_comppsc"),
            r"$kT_{\rm e}$ (keV)",
            r"$l_{\rm diss}$",
            r"Pair balance: $l_{\rm diss}$-$kT_{\rm e}$",
        ),
        (
            axes[2],
            ("tau_T_compps", "tau_T_comppsc"),
            ("l_diss_compps", "l_diss_comppsc"),
            r"$\tau_{\rm T}$",
            r"$l_{\rm diss}$",
            r"Pair line: $l_{\rm diss}$-$\tau_{\rm T}$",
        ),
    )

    for d_ratio, color in COLORS.items():
        selected = curve(rows, d_ratio)
        for ax, x_keys, y_keys, _, _, _ in panels:
            if isinstance(x_keys, tuple):
                x_old = [row[x_keys[0]] for row in selected]
                x_new = [row[x_keys[1]] for row in selected]
            else:
                x_old = x_new = [row[x_keys] for row in selected]
            ax.plot(
                x_old,
                [row[y_keys[0]] for row in selected],
                color=color,
                linestyle="--",
                linewidth=1.9,
            )
            ax.plot(
                x_new,
                [row[y_keys[1]] for row in selected],
                color=color,
                linestyle="-",
                linewidth=2.1,
            )

    for ax, _, _, xlabel, ylabel, title in panels:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11.5)

    axes[1].set_ylim(1.0e1, 1.0e6)
    axes[2].set_ylim(1.0e1, 1.0e6)

    color_handles = [
        Line2D([0], [0], color=color, lw=2.2, label=f"d = {d_ratio:g}")
        for d_ratio, color in COLORS.items()
    ]
    model_handles = [
        Line2D([0], [0], color="#222222", lw=2.2, linestyle="-", label="compPSc"),
        Line2D([0], [0], color="#222222", lw=2.0, linestyle="--", label="compPS"),
    ]
    first_legend = axes[0].legend(handles=color_handles, loc="lower left", title="Disk ratio")
    axes[0].add_artist(first_legend)
    axes[0].legend(handles=model_handles, loc="upper right", title="Transfer model")

    fig.suptitle("Slab Pair-Balance Curves: compPS versus compPSc", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(COMPARISON_PNG, dpi=200)
    plt.close(fig)


def plot_ratios(rows: list[dict[str, float]]) -> None:
    fig, (ax_tau, ax_ldiss) = plt.subplots(2, 1, figsize=(8.6, 8.4), sharex=True)

    for d_ratio, color in COLORS.items():
        selected = curve(rows, d_ratio)
        kte = [row["kTe_keV"] for row in selected]
        ax_tau.plot(
            kte,
            [100.0 * row["tau_fractional_difference"] for row in selected],
            color=color,
            lw=2.0,
            label=f"d = {d_ratio:g}",
        )
        ax_ldiss.plot(
            kte,
            [row["l_diss_ratio"] for row in selected],
            color=color,
            lw=2.0,
        )

    ax_tau.axhline(0.0, color="#555555", lw=1.0)
    ax_tau.axhspan(-8.0, 8.0, color="#777777", alpha=0.10)
    ax_tau.set_ylabel(r"$100(\tau_{\rm PSc}/\tau_{\rm PS}-1)$ (%)")
    ax_tau.set_title("Matched-Temperature Differences")
    ax_tau.legend(ncol=2)

    ax_ldiss.axhline(1.0, color="#555555", lw=1.0)
    ax_ldiss.set_xscale("log")
    ax_ldiss.set_yscale("log")
    ax_ldiss.set_xlabel(r"$kT_{\rm e}$ (keV)")
    ax_ldiss.set_ylabel(r"$l_{\rm diss,PSc}/l_{\rm diss,PS}$")

    fig.tight_layout()
    fig.savefig(RATIO_PNG, dpi=200)
    plt.close(fig)


def write_matched_rows(rows: list[dict[str, float]]) -> None:
    MATCHED_CSV.parent.mkdir(exist_ok=True)
    with MATCHED_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, float]]) -> None:
    print("d_ratio  |tau diff| median/p95/max   |log10(l ratio)| median/p95/max")
    for d_ratio in COLORS:
        selected = curve(rows, d_ratio)
        tau_abs = np.abs([row["tau_fractional_difference"] for row in selected])
        ldiss_log_abs = np.abs(np.log10([row["l_diss_ratio"] for row in selected]))
        print(
            f"{d_ratio:7g}  "
            f"{np.median(tau_abs):.4f}/{np.percentile(tau_abs, 95):.4f}/{np.max(tau_abs):.4f}   "
            f"{np.median(ldiss_log_abs):.4f}/{np.percentile(ldiss_log_abs, 95):.4f}/{np.max(ldiss_log_abs):.4f}"
        )


def main() -> None:
    configure_style()
    OUTPUT.mkdir(exist_ok=True)
    rows = match_rows(load_rows(COMPPS_CSV), load_rows(COMPPSC_CSV))
    if not rows:
        raise RuntimeError("No matched rows found.")
    write_matched_rows(rows)
    plot_comparison(rows)
    plot_ratios(rows)
    print_summary(rows)
    print(COMPARISON_PNG)
    print(RATIO_PNG)
    print(MATCHED_CSV)


if __name__ == "__main__":
    main()
