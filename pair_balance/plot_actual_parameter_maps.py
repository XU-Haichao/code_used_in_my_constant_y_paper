#!/usr/bin/env python3
"""Plot actual compPSc/ireflect interpolation-parameter maps."""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
from dataclasses import dataclass

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pair_balance" / "data"
OUTPUT_DIR = ROOT / "output"
PARAMETER_MAP_CMAP = "viridis"
Y_RIDGE_BAND_ALPHA = 0.16
DOUBLE_COLUMN_FIGSIZE = (7.2, 6.0)


@dataclass(frozen=True)
class MatrixTable:
    tau_values: np.ndarray
    kTe_values: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class PlotSpec:
    key: str
    path: pathlib.Path
    title: str
    colorbar_label: str
    cmap: str
    log_norm: bool
    vmin: float | None = None
    vmax: float | None = None
    extend: str = "neither"


@dataclass(frozen=True)
class YRidgeStatistics:
    mean_y: float
    sigma_log_y: float
    n_points: int


def column_suffix(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def parse_kte_header(header: str) -> float:
    if "_kTe_" not in header:
        raise ValueError(f"Cannot parse kTe from header {header!r}")
    suffix = header.rsplit("_kTe_", maxsplit=1)[1]
    return float(suffix.replace("p", ".").replace("m", "-"))


def geometric_edges(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(values, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("geometric edges need at least two centers")
    if np.any(centers <= 0.0) or np.any(np.diff(centers) <= 0.0):
        raise ValueError("centers must be positive and strictly increasing")
    log_centers = np.log(centers)
    log_edges = np.empty(centers.size + 1, dtype=float)
    log_edges[1:-1] = 0.5 * (log_centers[:-1] + log_centers[1:])
    log_edges[0] = 2.0 * log_centers[0] - log_edges[1]
    log_edges[-1] = 2.0 * log_centers[-1] - log_edges[-2]
    return np.exp(log_edges)


def geometric_edges_with_lower_bound(values: np.ndarray, *, lower_bound: float) -> np.ndarray:
    edges = geometric_edges(values)
    edges[0] = max(edges[0], float(lower_bound))
    return edges


def mask_unconverged(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=float)
    if values.shape != valid.shape:
        raise ValueError(f"value shape {values.shape} does not match valid shape {valid.shape}")
    return np.where(valid >= 0.5, values, np.nan)


def restrict_tau_range(table: MatrixTable, *, tau_min: float) -> MatrixTable:
    if tau_min <= 0.0:
        return table
    keep = table.tau_values >= float(tau_min)
    if not np.any(keep):
        raise ValueError(f"No tau grid points remain above tau_min={tau_min:g}")
    return MatrixTable(
        tau_values=table.tau_values[keep],
        kTe_values=table.kTe_values,
        values=table.values[keep, :],
    )


def theta_from_kte(kte_keV: np.ndarray | float) -> np.ndarray:
    return np.asarray(kte_keV, dtype=float) / 511.0


def bessel_factor(theta: np.ndarray | float) -> np.ndarray:
    from scipy.special import kve

    theta = np.asarray(theta, dtype=float)
    x = 1.0 / theta
    return 4.0 * theta * kve(3, x) / kve(2, x)


def tau_curve_from_bessel_y(kTe_values: np.ndarray, mean_y: float) -> np.ndarray:
    return float(mean_y) / bessel_factor(theta_from_kte(kTe_values))


def tau_band_from_bessel_y(kTe_values: np.ndarray, *, mean_y: float, sigma_log_y: float) -> tuple[np.ndarray, np.ndarray]:
    lower = tau_curve_from_bessel_y(kTe_values, float(mean_y) * 10.0 ** (-float(sigma_log_y)))
    upper = tau_curve_from_bessel_y(kTe_values, float(mean_y) * 10.0 ** float(sigma_log_y))
    return np.minimum(lower, upper), np.maximum(lower, upper)


def compute_compTT_main_y_statistics(frame) -> YRidgeStatistics:
    import pandas as pd

    compTT_plot = frame.copy()
    numeric_cols = [
        "Electron_Temperature_keV",
        "Optical_Depth_tau",
        "Eddington_Ratio",
    ]
    compTT_plot[numeric_cols] = compTT_plot[numeric_cols].apply(pd.to_numeric, errors="coerce")
    compTT_plot["Optical_Depth_tau"] = 2.0 * compTT_plot["Optical_Depth_tau"]
    compTT_plot["force_gray"] = compTT_plot["Source"].eq("NGC 5506")
    compTT_plot["in_lambda_range"] = compTT_plot["Eddington_Ratio"].between(0.01, 1.0, inclusive="both")
    compTT_plot["use_lambda_color"] = compTT_plot["in_lambda_range"] & ~compTT_plot["force_gray"]
    compTT_plot["is_mcg_5_23_16"] = compTT_plot["Source"].eq("MCG-5-23-16")

    compTT_y_sample = compTT_plot.loc[
        compTT_plot["use_lambda_color"] & ~compTT_plot["is_mcg_5_23_16"]
    ].copy()
    theta_main = theta_from_kte(compTT_y_sample["Electron_Temperature_keV"].to_numpy(dtype=float))
    tau_main = compTT_y_sample["Optical_Depth_tau"].to_numpy(dtype=float)
    y_main = np.asarray(bessel_factor(theta_main) * tau_main, dtype=float)
    y_main = y_main[np.isfinite(y_main) & (y_main > 0.0)]
    if y_main.size == 0:
        raise ValueError("No valid compTT main-sample points remain for y statistics")
    mean_y = float(np.mean(y_main))
    sigma_log_y = float(np.std(np.log10(y_main) - np.log10(mean_y)))
    return YRidgeStatistics(mean_y=mean_y, sigma_log_y=sigma_log_y, n_points=int(y_main.size))


def load_compTT_main_y_statistics(path: pathlib.Path) -> YRidgeStatistics:
    import pandas as pd

    return compute_compTT_main_y_statistics(pd.read_csv(path))


def load_matrix_csv(path: pathlib.Path) -> MatrixTable:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        kTe_values = np.array([parse_kte_header(cell) for cell in header[1:]], dtype=float)
        tau_values: list[float] = []
        rows: list[list[float]] = []
        for row in reader:
            tau_values.append(float(row[0]))
            rows.append([float(cell) if cell else math.nan for cell in row[1:]])
    values = np.array(rows, dtype=float)
    return MatrixTable(tau_values=np.array(tau_values, dtype=float), kTe_values=kTe_values, values=values)


def actual_matrix_paths(*, fixed_xi: float, grid_tag: str) -> tuple[dict[str, pathlib.Path], pathlib.Path]:
    slug = f"xi{column_suffix(fixed_xi)}_{grid_tag}"
    paths = {
        "eta": DATA_DIR / f"comppsc_actual_eta_{slug}_matrix.csv",
        "A_model": DATA_DIR / f"comppsc_actual_A_model_{slug}_matrix.csv",
        "p_sc": DATA_DIR / f"comppsc_actual_p_sc_{slug}_matrix.csv",
        "albedo": DATA_DIR / f"comppsc_actual_albedo_ireflect_{slug}_matrix.csv",
    }
    valid_path = DATA_DIR / f"comppsc_actual_valid_{slug}_matrix.csv"
    return paths, valid_path


def parameter_plot_specs(*, fixed_xi: float, paths: dict[str, pathlib.Path]) -> list[PlotSpec]:
    return [
        PlotSpec(
            key="eta",
            path=paths.get("eta", pathlib.Path()),
            title=r"$\eta$",
            colorbar_label=r"$\eta$",
            cmap=PARAMETER_MAP_CMAP,
            log_norm=False,
        ),
        PlotSpec(
            key="A_model",
            path=paths.get("A_model", pathlib.Path()),
            title=r"$A$",
            colorbar_label=r"$A$",
            cmap=PARAMETER_MAP_CMAP,
            log_norm=True,
            vmin=0.1,
            vmax=500.0,
            extend="both",
        ),
        PlotSpec(
            key="p_sc",
            path=paths.get("p_sc", pathlib.Path()),
            title=r"$p_{\rm sc}$",
            colorbar_label=r"$p_{\rm sc}$",
            cmap=PARAMETER_MAP_CMAP,
            log_norm=False,
        ),
        PlotSpec(
            key="albedo",
            path=paths.get("albedo", pathlib.Path()),
            title=rf"$a$, ireflect $\xi={fixed_xi:g}$",
            colorbar_label=r"$a$",
            cmap=PARAMETER_MAP_CMAP,
            log_norm=True,
            vmin=1.0e-3,
            extend="min",
        ),
    ]


def plot_parameter_maps(*, fixed_xi: float, grid_tag: str, tau_min_plot: float = 0.2) -> tuple[pathlib.Path, pathlib.Path]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_comppsc_actual_parameter_maps")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    paths, valid_path = actual_matrix_paths(fixed_xi=fixed_xi, grid_tag=grid_tag)
    valid_table = restrict_tau_range(load_matrix_csv(valid_path), tau_min=tau_min_plot)
    y_stats = load_compTT_main_y_statistics(ROOT / "data" / "compTT_comptt_sample_v2.csv")
    x_curve = np.logspace(np.log10(valid_table.kTe_values[0]), np.log10(valid_table.kTe_values[-1]), 500)
    tau_y_curve = tau_curve_from_bessel_y(x_curve, y_stats.mean_y)
    tau_y_lower, tau_y_upper = tau_band_from_bessel_y(
        x_curve,
        mean_y=y_stats.mean_y,
        sigma_log_y=y_stats.sigma_log_y,
    )
    specs = parameter_plot_specs(fixed_xi=fixed_xi, paths=paths)

    x_edges = geometric_edges(valid_table.kTe_values)
    y_edges = geometric_edges_with_lower_bound(valid_table.tau_values, lower_bound=tau_min_plot)

    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linestyle": ":",
            "axes.facecolor": "#f8f8f5",
            "figure.facecolor": "white",
            "font.size": 10.5,
            "mathtext.fontset": "dejavuserif",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=DOUBLE_COLUMN_FIGSIZE, sharex=True, sharey=True, constrained_layout=True)

    for ax, spec in zip(axes.ravel(), specs, strict=True):
        table = restrict_tau_range(load_matrix_csv(spec.path), tau_min=tau_min_plot)
        if not np.allclose(table.kTe_values, valid_table.kTe_values) or not np.allclose(
            table.tau_values, valid_table.tau_values
        ):
            raise ValueError(f"{spec.path} grid does not match {valid_path}")
        values = mask_unconverged(table.values, valid_table.values)
        positive = values[np.isfinite(values) & (values > 0.0)]
        cmap = plt.colormaps[spec.cmap].copy()
        cmap.set_bad(color="#d8d8d8")
        norm = None
        if spec.log_norm:
            vmin = float(spec.vmin) if spec.vmin is not None else float(np.min(positive))
            vmax = float(spec.vmax) if spec.vmax is not None else float(np.max(positive))
            norm = LogNorm(vmin=vmin, vmax=vmax)
            cmap.set_under(cmap(0.0))
            cmap.set_over(cmap(1.0))
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_invalid(values),
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        valid_band = (
            np.isfinite(tau_y_lower)
            & np.isfinite(tau_y_upper)
            & (tau_y_upper >= float(tau_min_plot))
            & (tau_y_lower > 0.0)
            & (tau_y_upper > 0.0)
        )
        ax.fill_between(
            x_curve[valid_band],
            np.maximum(tau_y_lower[valid_band], float(tau_min_plot)),
            tau_y_upper[valid_band],
            color="black",
            alpha=Y_RIDGE_BAND_ALPHA,
            linewidth=0,
            zorder=7,
        )
        valid_curve = np.isfinite(tau_y_curve) & (tau_y_curve >= float(tau_min_plot))
        ax.plot(
            x_curve[valid_curve],
            tau_y_curve[valid_curve],
            color="black",
            linestyle="-",
            linewidth=1.35,
            alpha=0.95,
            label=fr"$\langle y\rangle={y_stats.mean_y:.3f}$",
            zorder=8,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(spec.title)
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(float(tau_min_plot), y_edges[-1])
        fig.colorbar(mesh, ax=ax, pad=0.015, shrink=0.92, extend=spec.extend)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$kT_{\rm e}$ (keV)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\tau_{\rm T}$")
    for ax in axes[:, 1]:
        ax.tick_params(axis="y", which="both", labelleft=True)
        ax.set_ylabel(r"$\tau_{\rm T}$")

    axes[0, 0].legend(loc="lower right", frameon=True, framealpha=0.75, fontsize=8.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"comppsc_actual_parameter_maps_xi{column_suffix(fixed_xi)}_{grid_tag}.png"
    pdf_path = OUTPUT_DIR / f"comppsc_actual_parameter_maps_xi{column_suffix(fixed_xi)}_{grid_tag}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-xi", type=float, default=100.0)
    parser.add_argument("--grid-tag", default="log32x32")
    parser.add_argument("--tau-min-plot", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    png_path, pdf_path = plot_parameter_maps(
        fixed_xi=args.fixed_xi,
        grid_tag=args.grid_tag,
        tau_min_plot=args.tau_min_plot,
    )
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
