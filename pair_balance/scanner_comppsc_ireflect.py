#!/usr/bin/env python3
"""One-pass compPSc energy balance using XSPEC ireflect."""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import sys
from dataclasses import dataclass

import numpy as np


ROOT_FOR_IMPORT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from pair_balance.scanner_comppsc import ComppscScanConfig, ComppscSlabSolver
from pair_balance.scanner_reflect import IonizedReflectionConfig, IonizedReflectionKernel


EV_TO_KELVIN = 11604.518121550082
STEFAN_BOLTZMANN_CGS = 5.670374419e-5
MEC2_KEV = 511.0


@dataclass(frozen=True)
class IreflectComppscConfig:
    kTe_kev: float = 10.0
    tbb_kev: float = 0.005
    density_cm3: float = 1.0e15
    max_scatter: int = 2000
    exact_angles: bool = True
    observer_mu: float = 0.5
    hemisphere_mu_order: int = 12
    reflector_abundance: float = 1.0
    reflector_iron_abundance: float = 1.0
    energy_min_kev: float = 0.005
    energy_max_kev: float = 20.0
    tau_min: float = 0.5
    tau_max: float = 5.0
    root_tolerance: float = 5.0e-4
    root_iterations: int = 40
    global_tau_samples: int = 8


def ev_to_kelvin(temperature_ev: float) -> float:
    if temperature_ev <= 0.0:
        raise ValueError("disk temperature must be positive")
    return float(temperature_ev) * EV_TO_KELVIN


def blackbody_surface_flux(temperature_k: float) -> float:
    if temperature_k <= 0.0:
        raise ValueError("disk temperature must be positive")
    return STEFAN_BOLTZMANN_CGS * float(temperature_k) ** 4


def band_energy_flux(
    x_grid: np.ndarray,
    x_weights: np.ndarray,
    spectrum: np.ndarray,
    *,
    energy_min_kev: float,
    energy_max_kev: float,
) -> float:
    x = np.asarray(x_grid, dtype=float)
    weights = np.asarray(x_weights, dtype=float)
    values = np.asarray(spectrum, dtype=float)
    if x.shape != weights.shape or x.shape != values.shape:
        raise ValueError("x_grid, x_weights, and spectrum must have identical shapes")
    if x.ndim != 1 or x.size < 2 or np.any(x <= 0.0) or np.any(np.diff(x) <= 0.0):
        raise ValueError("x_grid must be a strictly increasing positive array")
    if energy_min_kev < 0.0 or energy_max_kev <= energy_min_kev:
        raise ValueError("energy band must satisfy 0 <= min < max")

    log_centers = np.log(MEC2_KEV * x)
    log_edges = np.empty(x.size + 1, dtype=float)
    log_edges[1:-1] = 0.5 * (log_centers[:-1] + log_centers[1:])
    log_edges[0] = 2.0 * log_centers[0] - log_edges[1]
    log_edges[-1] = 2.0 * log_centers[-1] - log_edges[-2]
    log_min = -math.inf if energy_min_kev == 0.0 else math.log(energy_min_kev)
    log_max = math.inf if math.isinf(energy_max_kev) else math.log(energy_max_kev)
    overlap = np.maximum(
        0.0,
        np.minimum(log_edges[1:], log_max) - np.maximum(log_edges[:-1], log_min),
    )
    fraction = overlap / np.diff(log_edges)
    return float(np.sum(x * weights * values * fraction))


def ionization_parameter(ionizing_flux_cgs: float, *, density_cm3: float) -> float:
    if ionizing_flux_cgs < 0.0:
        raise ValueError("ionizing flux cannot be negative")
    if density_cm3 <= 0.0:
        raise ValueError("reflector density must be positive")
    return 4.0 * math.pi * float(ionizing_flux_cgs) / float(density_cm3)


class IreflectComppscSolver:
    def __init__(self, config: IreflectComppscConfig):
        self.config = config
        theta = config.kTe_kev / MEC2_KEV
        self.theta = theta
        self.disk_temperature_k = ev_to_kelvin(1000.0 * config.tbb_kev)
        self.seed_surface_flux_cgs = blackbody_surface_flux(self.disk_temperature_k)
        transfer_config = ComppscScanConfig(
            theta_min=theta,
            theta_max=theta,
            n_samples=1,
            tbb_kev=config.tbb_kev,
            max_scatter=config.max_scatter,
            exact_angles=config.exact_angles,
            observer_mu=config.observer_mu,
            tau_min=config.tau_min,
            tau_max=config.tau_max,
        )
        self.transfer = ComppscSlabSolver(transfer_config)
        self._cache: dict[float, dict[str, float | int]] = {}

    def evaluate_tau(self, tau_t: float) -> dict[str, float | int]:
        key = round(float(tau_t), 12)
        if key in self._cache:
            return self._cache[key]

        state = self.transfer.run_state(self.theta, tau_t)
        last_scatter = int(getattr(self.transfer.module.msiterstat, "lastisc", -1))
        last_difmax = float(getattr(self.transfer.module.msiterstat, "lastdifmax", math.nan))
        ionizing_flux_model = band_energy_flux(
            state.x_grid,
            state.x_weights,
            state.comp_down_spectrum_model,
            energy_min_kev=self.config.energy_min_kev,
            energy_max_kev=self.config.energy_max_kev,
        )
        ionizing_flux_cgs = (
            ionizing_flux_model
            / max(state.seed_flux_model, 1.0e-30)
            * self.seed_surface_flux_cgs
        )
        xi = ionization_parameter(ionizing_flux_cgs, density_cm3=self.config.density_cm3)
        reflector = IonizedReflectionKernel(
            IonizedReflectionConfig(
                disk_temperature_k=self.disk_temperature_k,
                ionization_parameter=xi,
                reflector_abundance=self.config.reflector_abundance,
                reflector_iron_abundance=self.config.reflector_iron_abundance,
                hemisphere_mu_order=self.config.hemisphere_mu_order,
            )
        )
        observer_spectrum, _, observer_flux, hemisphere_flux = reflector.hemisphere_response(
            state.x_grid,
            state.x_weights,
            state.comp_down_spectrum_model,
            self.config.observer_mu,
        )
        effective_albedo = hemisphere_flux / max(state.comp_down_flux_model, 1.0e-30)
        effective_albedo = min(max(effective_albedo, 0.0), 0.999999)
        amplification_required = 1.0 / max(
            (1.0 - effective_albedo) * state.eta,
            1.0e-30,
        )
        residual = math.log(state.amplification_model / amplification_required)
        result: dict[str, float | int] = {
            "kTe_keV": self.config.kTe_kev,
            "theta": self.theta,
            "tau_T": tau_t,
            "density_cm3": self.config.density_cm3,
            "disk_temperature_K": self.disk_temperature_k,
            "seed_surface_flux_cgs": self.seed_surface_flux_cgs,
            "ionizing_flux_model": ionizing_flux_model,
            "ionizing_flux_cgs": ionizing_flux_cgs,
            "ionization_parameter": xi,
            "effective_albedo": effective_albedo,
            "eta": state.eta,
            "p_sc": state.p_sc,
            "A_model": state.amplification_model,
            "A_required": amplification_required,
            "energy_log_residual": residual,
            "seed_flux_model": state.seed_flux_model,
            "downward_flux_model": state.comp_down_flux_model,
            "reflected_flux_model_observer": observer_flux,
            "reflected_flux_model_hemisphere": hemisphere_flux,
            "observer_reflection_peak_x": float(state.x_grid[int(np.argmax(observer_spectrum))]),
            "last_scatter_order": last_scatter,
            "last_difmax": last_difmax,
        }
        self._cache[key] = result
        return result

    def solve_tau(self) -> dict[str, float | int | str]:
        tau_grid = np.geomspace(
            self.config.tau_min,
            self.config.tau_max,
            self.config.global_tau_samples,
        )
        sampled = [self.evaluate_tau(float(tau_t)) for tau_t in tau_grid]
        bracket: tuple[float, float] | None = None
        for lower, upper in zip(sampled[:-1], sampled[1:]):
            f_lower = float(lower["energy_log_residual"])
            f_upper = float(upper["energy_log_residual"])
            if f_lower * f_upper <= 0.0:
                bracket = (float(lower["tau_T"]), float(upper["tau_T"]))
        if bracket is None:
            raise RuntimeError(
                "No ireflect energy-balance root found in "
                f"tau=[{self.config.tau_min:g}, {self.config.tau_max:g}]"
            )

        tau_lo, tau_hi = bracket
        f_lo = float(self.evaluate_tau(tau_lo)["energy_log_residual"])
        root_iterations = 0
        for root_iterations in range(1, self.config.root_iterations + 1):
            tau_mid = math.sqrt(tau_lo * tau_hi)
            middle = self.evaluate_tau(tau_mid)
            f_mid = float(middle["energy_log_residual"])
            if abs(f_mid) <= self.config.root_tolerance:
                result = dict(middle)
                result["root_method"] = "log_bisection"
                result["root_iterations"] = root_iterations
                return result
            if f_lo * f_mid <= 0.0:
                tau_hi = tau_mid
            else:
                tau_lo = tau_mid
                f_lo = f_mid

        result = dict(self.evaluate_tau(math.sqrt(tau_lo * tau_hi)))
        result["root_method"] = "log_bisection_limit"
        result["root_iterations"] = root_iterations
        return result

    def evaluated_rows(self) -> list[dict[str, float | int]]:
        return [self._cache[key] for key in sorted(self._cache)]


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pair_balance" / "data"
OUTPUT_DIR = ROOT / "output"
SUMMARY_CSV = DATA_DIR / "comppsc_ireflect_10kev_density_compare.csv"
SCAN_CSV = DATA_DIR / "comppsc_ireflect_10kev_tau_scan.csv"
COMPARISON_PNG = OUTPUT_DIR / "comppsc_ireflect_10kev_density_compare.png"


def write_rows(path: pathlib.Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_density_comparison(
    scan_rows: list[dict[str, float | int]],
    roots: list[dict[str, float | int | str]],
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_comppsc_ireflect")
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
            "font.size": 10.5,
        }
    )
    palette = {1.0e13: "#D55E00", 1.0e15: "#0072B2"}
    fig, (ax_residual, ax_xi, ax_albedo) = plt.subplots(1, 3, figsize=(15.8, 5.0))

    for density in sorted({float(row["density_cm3"]) for row in scan_rows}):
        selected = sorted(
            (row for row in scan_rows if float(row["density_cm3"]) == density),
            key=lambda row: float(row["tau_T"]),
        )
        color = palette.get(density, "#555555")
        density_exponent = int(round(math.log10(density)))
        label = rf"$n=10^{{{density_exponent}}}\,\mathrm{{cm}}^{{-3}}$"
        tau = [float(row["tau_T"]) for row in selected]
        ax_residual.plot(
            tau,
            [float(row["energy_log_residual"]) for row in selected],
            color=color,
            lw=2.0,
            marker="o",
            markersize=3.5,
            label=label,
        )
        ax_xi.plot(
            tau,
            [float(row["ionization_parameter"]) for row in selected],
            color=color,
            lw=2.0,
            marker="o",
            markersize=3.5,
        )
        ax_albedo.plot(
            tau,
            [float(row["effective_albedo"]) for row in selected],
            color=color,
            lw=2.0,
            marker="o",
            markersize=3.5,
        )

    ax_residual.axhline(0.0, color="#222222", lw=1.0)
    for root in roots:
        density = float(root["density_cm3"])
        ax_residual.axvline(
            float(root["tau_T"]),
            color=palette.get(density, "#555555"),
            ls="--",
            lw=1.4,
        )
    ax_residual.set_xlabel(r"$\tau_{\rm T}$")
    ax_residual.set_ylabel(r"$\ln(A_{\rm model}/A_{\rm required})$")
    ax_residual.set_title("One-pass energy-balance root")
    ax_residual.legend()

    ax_xi.set_yscale("log")
    ax_xi.set_xlabel(r"$\tau_{\rm T}$")
    ax_xi.set_ylabel(r"$\xi$ (erg cm s$^{-1}$)")
    ax_xi.set_title(r"$\xi=4\pi F_{5\,\rm eV-20\,keV}/n$")

    ax_albedo.set_yscale("log")
    ax_albedo.set_xlabel(r"$\tau_{\rm T}$")
    ax_albedo.set_ylabel("Effective ireflect albedo")
    ax_albedo.set_title("Spectrum-dependent disk response")

    fig.suptitle(r"compPSc + ireflect at $kT_{\rm e}=10$ keV, $kT_{\rm disk}=5$ eV")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPARISON_PNG, dpi=200)
    plt.close(fig)


def run_density_comparison(
    *,
    kTe_kev: float,
    densities_cm3: list[float],
    max_scatter: int,
    hemisphere_mu_order: int,
    tau_min: float,
    tau_max: float,
) -> list[dict[str, float | int | str]]:
    roots: list[dict[str, float | int | str]] = []
    scan_rows: list[dict[str, float | int]] = []
    for density in densities_cm3:
        solver = IreflectComppscSolver(
            IreflectComppscConfig(
                kTe_kev=kTe_kev,
                density_cm3=density,
                max_scatter=max_scatter,
                hemisphere_mu_order=hemisphere_mu_order,
                tau_min=tau_min,
                tau_max=tau_max,
            )
        )
        root = solver.solve_tau()
        roots.append(root)
        scan_rows.extend(solver.evaluated_rows())
        print(
            f"n={density:.3e} tau={float(root['tau_T']):.7f} "
            f"xi={float(root['ionization_parameter']):.6g} "
            f"albedo={float(root['effective_albedo']):.6g} "
            f"residual={float(root['energy_log_residual']):.3g}",
            flush=True,
        )

    write_rows(SUMMARY_CSV, roots)
    write_rows(SCAN_CSV, scan_rows)
    plot_density_comparison(scan_rows, roots)
    return roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kTe", type=float, default=10.0)
    parser.add_argument("--densities", type=float, nargs="+", default=[1.0e13, 1.0e15])
    parser.add_argument("--max-scatter", type=int, default=2000)
    parser.add_argument("--mu-order", type=int, default=8)
    parser.add_argument("--tau-min", type=float, default=2.0)
    parser.add_argument("--tau-max", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_density_comparison(
        kTe_kev=args.kTe,
        densities_cm3=[float(value) for value in args.densities],
        max_scatter=args.max_scatter,
        hemisphere_mu_order=args.mu_order,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
    )
    print(SUMMARY_CSV)
    print(SCAN_CSV)
    print(COMPARISON_PNG)


if __name__ == "__main__":
    main()
