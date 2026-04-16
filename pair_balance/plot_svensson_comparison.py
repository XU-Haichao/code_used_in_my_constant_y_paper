#!/usr/bin/env python3
"""Plot pair-line comparisons against calibrated Stern et al. figure points."""

from __future__ import annotations

import csv
import os
import pathlib
from typing import Iterable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_pair_balance_compare")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
PAIR_BALANCE = ROOT / "pair_balance"
PAIR_DATA = PAIR_BALANCE / "data"
CAL_DATA = ROOT / "data"
OUTPUT = ROOT / "output"

FIXED_SCAN = PAIR_DATA / "ps96_slab_pair_line_theta_0.02_1.0_log50.csv"
COMPARE_SCAN = PAIR_DATA / "ps96_slab_pair_line_reflection_compare_theta_0.02_1.0_log50.csv"
CAL_L_TAUT = CAL_DATA / "svensson_calibrated_fig1_l_tauT.csv"
CAL_L_THETA = CAL_DATA / "svensson_calibrated_fig1_l_theta.csv"


def load_csv_rows(path: pathlib.Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            parsed: dict[str, float | str] = {}
            for key, value in raw.items():
                if value is None or value == "":
                    parsed = {}
                    break
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            if parsed:
                rows.append(parsed)
    return rows


def sort_xy(x: Iterable[float], y: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    order = np.argsort(x_arr)
    return x_arr[order], y_arr[order]


def interp_logx(model_x: np.ndarray, model_y: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    return np.interp(np.log10(target_x), np.log10(model_x), model_y)


def make_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (8, 6),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "axes.facecolor": "#fbfbf8",
            "figure.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 11,
            "legend.frameon": False,
        }
    )


def plot_l_tau(compare_rows: list[dict[str, float]], cal_tau_rows: list[dict[str, float]]) -> pathlib.Path:
    output = OUTPUT / "svensson_compare_l_tauT_loglog.png"
    fig, ax = plt.subplots()

    fixed_l, fixed_tau = sort_xy((row["l_diss_fixed"] for row in compare_rows), (row["tau_T_fixed"] for row in compare_rows))
    refl_l, refl_tau = sort_xy((row["l_diss_reflect"] for row in compare_rows), (row["tau_T_reflect"] for row in compare_rows))
    cal_l, cal_tau = sort_xy((row["ldiss"] for row in cal_tau_rows), (row["tau_T"] for row in cal_tau_rows))

    ax.plot(fixed_l, fixed_tau, color="#1f77b4", lw=2.2, label="Fixed albedo = 0.2")
    ax.plot(refl_l, refl_tau, color="#d95f02", lw=2.2, label="Reflection-coupled")
    ax.scatter(cal_l, cal_tau, color="#222222", s=24, alpha=0.9, label="Calibrated Stern+95 fig.1")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$l_{\rm diss}$")
    ax.set_ylabel(r"$\tau_{\rm T}$")
    ax.set_title(r"Log-Log Pair Line Comparison in $l_{\rm diss}-\tau_{\rm T}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_l_theta(compare_rows: list[dict[str, float]], cal_theta_rows: list[dict[str, float]]) -> pathlib.Path:
    output = OUTPUT / "svensson_compare_l_theta_loglog.png"
    fig, ax = plt.subplots()

    fixed_l, fixed_theta = sort_xy((row["l_diss_fixed"] for row in compare_rows), (row["theta"] for row in compare_rows))
    refl_l, refl_theta = sort_xy((row["l_diss_reflect"] for row in compare_rows), (row["theta"] for row in compare_rows))
    cal_l, cal_theta = sort_xy((row["ldiss"] for row in cal_theta_rows), (row["theta"] for row in cal_theta_rows))

    ax.plot(fixed_l, fixed_theta, color="#1f77b4", lw=2.2, label="Fixed albedo = 0.2")
    ax.plot(refl_l, refl_theta, color="#d95f02", lw=2.2, label="Reflection-coupled")
    ax.scatter(cal_l, cal_theta, color="#222222", s=24, alpha=0.9, label="Calibrated Stern+95 fig.1")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$l_{\rm diss}$")
    ax.set_ylabel(r"$\Theta$")
    ax.set_title(r"Log-Log Pair Line Comparison in $l_{\rm diss}-\Theta$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_residuals(compare_rows: list[dict[str, float]], cal_tau_rows: list[dict[str, float]], cal_theta_rows: list[dict[str, float]]) -> pathlib.Path:
    output = OUTPUT / "svensson_compare_residuals.png"
    fig, (ax_tau, ax_theta) = plt.subplots(2, 1, figsize=(8, 9), sharex=False)

    fixed_l_tau, fixed_tau = sort_xy((row["l_diss_fixed"] for row in compare_rows), (row["tau_T_fixed"] for row in compare_rows))
    refl_l_tau, refl_tau = sort_xy((row["l_diss_reflect"] for row in compare_rows), (row["tau_T_reflect"] for row in compare_rows))
    cal_l_tau, cal_tau = sort_xy((row["ldiss"] for row in cal_tau_rows), (row["tau_T"] for row in cal_tau_rows))

    fixed_interp_tau = interp_logx(fixed_l_tau, fixed_tau, cal_l_tau)
    refl_interp_tau = interp_logx(refl_l_tau, refl_tau, cal_l_tau)

    ax_tau.axhline(0.0, color="#666666", lw=1.0)
    ax_tau.plot(cal_l_tau, fixed_interp_tau / cal_tau - 1.0, color="#1f77b4", lw=2.0, label="Fixed albedo = 0.2")
    ax_tau.plot(cal_l_tau, refl_interp_tau / cal_tau - 1.0, color="#d95f02", lw=2.0, label="Reflection-coupled")
    ax_tau.set_xscale("log")
    ax_tau.set_ylabel(r"$\tau_{\rm T}$ frac. diff.")
    ax_tau.set_title("Fractional Difference Relative to Calibrated Points")
    ax_tau.legend()

    fixed_l_theta, fixed_theta = sort_xy((row["l_diss_fixed"] for row in compare_rows), (row["theta"] for row in compare_rows))
    refl_l_theta, refl_theta = sort_xy((row["l_diss_reflect"] for row in compare_rows), (row["theta"] for row in compare_rows))
    cal_l_theta, cal_theta = sort_xy((row["ldiss"] for row in cal_theta_rows), (row["theta"] for row in cal_theta_rows))

    fixed_interp_theta = interp_logx(fixed_l_theta, fixed_theta, cal_l_theta)
    refl_interp_theta = interp_logx(refl_l_theta, refl_theta, cal_l_theta)

    ax_theta.axhline(0.0, color="#666666", lw=1.0)
    ax_theta.plot(cal_l_theta, fixed_interp_theta / cal_theta - 1.0, color="#1f77b4", lw=2.0, label="Fixed albedo = 0.2")
    ax_theta.plot(cal_l_theta, refl_interp_theta / cal_theta - 1.0, color="#d95f02", lw=2.0, label="Reflection-coupled")
    ax_theta.set_xscale("log")
    ax_theta.set_xlabel(r"$l_{\rm diss}$")
    ax_theta.set_ylabel(r"$\Theta$ frac. diff.")

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    make_style()
    OUTPUT.mkdir(exist_ok=True)

    compare_rows = load_csv_rows(COMPARE_SCAN)
    cal_tau_rows = load_csv_rows(CAL_L_TAUT)
    cal_theta_rows = load_csv_rows(CAL_L_THETA)

    outputs = [
        plot_l_tau(compare_rows, cal_tau_rows),
        plot_l_theta(compare_rows, cal_theta_rows),
        plot_residuals(compare_rows, cal_tau_rows, cal_theta_rows),
    ]

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
