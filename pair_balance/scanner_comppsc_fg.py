#!/usr/bin/env python3
"""compPSc slab radiative-equilibrium curves for arbitrary f and g.

The closure follows the updated sandwich disk-corona model in the paper:

    A_req = (1 + p_sc d) / [g (1 - a) eta + d (1 - g a eta p_sc)],

where d=(1-f)/f.  The radiative-transfer quantities A_model, eta and p_sc are
computed from compPSc for a 5 eV blackbody seed spectrum.  The effective albedo
a is computed from XSPEC ireflect using the downward compPSc spectrum and a
fixed ionization parameter xi=100 by default.
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

from pair_balance.scanner import logspace
from pair_balance.scanner_comppsc import ComppscScanConfig, ComppscSlabSolver, RadiativeState
from pair_balance.scanner_comppsc_ireflect import ev_to_kelvin
from pair_balance.scanner_comppsc_eta_albedo_grid import column_suffix, effective_albedo
from pair_balance.scanner_reflect import IonizedReflectionConfig, IonizedReflectionKernel


MEC2_KEV = 511.0
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pair_balance" / "data"
OUTPUT_DIR = ROOT / "output"


@dataclass(frozen=True)
class CompactnessTermsFG:
    l_s_over_l_c: float
    l_h_over_l_c: float
    intrinsic_seed_over_l_c: float
    reprocessed_seed_over_l_c: float
    returning_reflection_over_l_c: float


@dataclass(frozen=True)
class FGEnergyBalanceConfig:
    kTe_min_kev: float = 10.0
    kTe_max_kev: float = 200.0
    n_samples: int = 24
    f_values: tuple[float, ...] = (1.0, 0.8, 0.5, 0.25)
    g_values: tuple[float, ...] = (1.0, 0.7, 0.5)
    fixed_xi: float = 100.0
    tbb_kev: float = 0.005
    tau_min: float = 0.03
    tau_max: float = 10.0
    max_scatter: int = 2000
    exact_angles: bool = True
    observer_mu: float = 0.5
    hemisphere_mu_order: int = 8
    root_tolerance: float = 5.0e-4
    root_iterations: int = 38
    global_tau_samples: int = 18
    continuation_expand_factor: float = 1.45
    continuation_expand_steps: int = 10


def d_ratio_from_f(f_corona: float) -> float:
    if not 0.0 < f_corona <= 1.0:
        raise ValueError("f must satisfy 0 < f <= 1")
    return (1.0 - float(f_corona)) / float(f_corona)


def f_from_d_ratio(d_ratio: float) -> float:
    if d_ratio < 0.0:
        raise ValueError("d_ratio cannot be negative")
    return 1.0 / (1.0 + float(d_ratio))


def validate_feedback_factor(feedback_factor: float) -> float:
    if not 0.0 <= feedback_factor <= 1.0:
        raise ValueError("g must satisfy 0 <= g <= 1")
    return float(feedback_factor)


def safe_positive(value: float, floor: float = 1.0e-30) -> float:
    return max(float(value), floor)


def amplification_required_fg(
    eta: float,
    p_sc: float,
    albedo: float,
    f_corona: float,
    feedback_factor: float,
) -> float:
    d_ratio = d_ratio_from_f(f_corona)
    g_factor = validate_feedback_factor(feedback_factor)
    denominator = g_factor * (1.0 - albedo) * eta + d_ratio * (1.0 - g_factor * albedo * eta * p_sc)
    return (1.0 + d_ratio * p_sc) / safe_positive(denominator)


def compactness_terms_fg(
    eta: float,
    p_sc: float,
    albedo: float,
    f_corona: float,
    feedback_factor: float,
) -> CompactnessTermsFG:
    d_ratio = d_ratio_from_f(f_corona)
    g_factor = validate_feedback_factor(feedback_factor)
    transport_denom = safe_positive(1.0 - p_sc * g_factor * eta)
    l_h_over_l_c = (1.0 + d_ratio * p_sc) / transport_denom
    intrinsic_seed = d_ratio
    reprocessed_seed = g_factor * (1.0 - albedo) * eta * l_h_over_l_c
    returning_reflection = g_factor * albedo * eta * l_h_over_l_c
    l_s_over_l_c = intrinsic_seed + reprocessed_seed
    return CompactnessTermsFG(
        l_s_over_l_c=l_s_over_l_c,
        l_h_over_l_c=l_h_over_l_c,
        intrinsic_seed_over_l_c=intrinsic_seed,
        reprocessed_seed_over_l_c=reprocessed_seed,
        returning_reflection_over_l_c=returning_reflection,
    )


def parse_float_list(values: list[str] | None, defaults: Sequence[float]) -> tuple[float, ...]:
    if not values:
        return tuple(float(value) for value in defaults)
    parsed: list[float] = []
    for value in values:
        parsed.extend(float(item) for item in value.split(",") if item)
    return tuple(parsed)


def output_paths(f_values: Sequence[float], g_values: Sequence[float], fixed_xi: float) -> tuple[pathlib.Path, pathlib.Path]:
    f_slug = "_".join(column_suffix(value) for value in f_values)
    g_slug = "_".join(column_suffix(value) for value in g_values)
    xi_slug = column_suffix(fixed_xi)
    return (
        DATA_DIR / f"comppsc_fg_equilibrium_xi{xi_slug}_f{f_slug}_g{g_slug}.csv",
        OUTPUT_DIR / f"comppsc_fg_equilibrium_xi{xi_slug}_f{f_slug}_g{g_slug}.png",
    )


class FGEnergyBalanceSolver:
    def __init__(self, config: FGEnergyBalanceConfig):
        self.config = config
        self.transfer = ComppscSlabSolver(
            ComppscScanConfig(
                theta_min=config.kTe_min_kev / MEC2_KEV,
                theta_max=config.kTe_max_kev / MEC2_KEV,
                n_samples=1,
                tbb_kev=config.tbb_kev,
                max_scatter=config.max_scatter,
                exact_angles=config.exact_angles,
                observer_mu=config.observer_mu,
                tau_min=config.tau_min,
                tau_max=config.tau_max,
            )
        )
        disk_temperature_k = ev_to_kelvin(1000.0 * config.tbb_kev)
        self.reflector = IonizedReflectionKernel(
            IonizedReflectionConfig(
                disk_temperature_k=disk_temperature_k,
                ionization_parameter=config.fixed_xi,
                hemisphere_mu_order=config.hemisphere_mu_order,
            )
        )
        self._state_albedo_cache: dict[tuple[float, float], tuple[RadiativeState, float, float, int, float]] = {}

    @staticmethod
    def _round_key(theta: float, tau_t: float) -> tuple[float, float]:
        return (round(float(theta), 12), round(float(tau_t), 12))

    def state_and_albedo(self, theta: float, tau_t: float) -> tuple[RadiativeState, float, float, int, float]:
        cache_key = self._round_key(theta, tau_t)
        if cache_key in self._state_albedo_cache:
            return self._state_albedo_cache[cache_key]

        state = self.transfer.run_state(theta, tau_t)
        last_scatter = int(getattr(self.transfer.module.msiterstat, "lastisc", -1))
        last_difmax = float(getattr(self.transfer.module.msiterstat, "lastdifmax", math.nan))
        _, _, _, hemisphere_flux = self.reflector.hemisphere_response(
            state.x_grid,
            state.x_weights,
            state.comp_down_spectrum_model,
            self.config.observer_mu,
        )
        albedo = effective_albedo(hemisphere_flux, state.comp_down_flux_model)
        result = (state, albedo, hemisphere_flux, last_scatter, last_difmax)
        self._state_albedo_cache[cache_key] = result
        return result

    def evaluate_tau(
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
        residual = math.log(state.amplification_model / a_required)
        converged = (
            last_scatter < self.config.max_scatter - 1
            and math.isfinite(last_difmax)
            and last_difmax <= 3.2e-3
        )
        return {
            "f_corona": f_corona,
            "g_feedback": feedback_factor,
            "d_ratio": d_ratio_from_f(f_corona),
            "theta": theta,
            "kTe_keV": theta * MEC2_KEV,
            "tau_T": tau_t,
            "fixed_xi": self.config.fixed_xi,
            "effective_albedo": albedo,
            "eta": state.eta,
            "p_sc": state.p_sc,
            "A_model": state.amplification_model,
            "A_required": a_required,
            "l_s_over_l_c": terms.l_s_over_l_c,
            "l_h_over_l_c": terms.l_h_over_l_c,
            "intrinsic_seed_over_l_c": terms.intrinsic_seed_over_l_c,
            "reprocessed_seed_over_l_c": terms.reprocessed_seed_over_l_c,
            "returning_reflection_over_l_c": terms.returning_reflection_over_l_c,
            "downward_flux_model": state.comp_down_flux_model,
            "reflected_flux_model_hemisphere": reflected_flux,
            "energy_log_residual": residual,
            "last_scatter_order": last_scatter,
            "last_difmax": last_difmax,
            "converged": converged,
            "max_scatter": self.config.max_scatter,
            "hemisphere_mu_order": self.config.hemisphere_mu_order,
            "exact_angles": self.config.exact_angles,
        }

    def energy_residual(self, theta: float, tau_t: float, *, f_corona: float, feedback_factor: float) -> float:
        return float(
            self.evaluate_tau(
                theta,
                tau_t,
                f_corona=f_corona,
                feedback_factor=feedback_factor,
            )["energy_log_residual"]
        )

    def _bisection(self, theta: float, lo: float, hi: float, *, f_corona: float, feedback_factor: float) -> tuple[float, int]:
        f_lo = self.energy_residual(theta, lo, f_corona=f_corona, feedback_factor=feedback_factor)
        f_hi = self.energy_residual(theta, hi, f_corona=f_corona, feedback_factor=feedback_factor)
        if f_lo * f_hi > 0.0:
            raise RuntimeError("Bisection requested without a sign change.")

        for iteration in range(1, self.config.root_iterations + 1):
            mid = math.sqrt(lo * hi)
            f_mid = self.energy_residual(theta, mid, f_corona=f_corona, feedback_factor=feedback_factor)
            if abs(f_mid) <= self.config.root_tolerance:
                return mid, iteration
            if f_lo * f_mid <= 0.0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid
        return math.sqrt(lo * hi), self.config.root_iterations

    def _global_roots(self, theta: float, *, f_corona: float, feedback_factor: float) -> list[tuple[float, int]]:
        tau_grid = np.geomspace(self.config.tau_min, self.config.tau_max, self.config.global_tau_samples)
        residuals = [
            self.energy_residual(theta, float(tau_t), f_corona=f_corona, feedback_factor=feedback_factor)
            for tau_t in tau_grid
        ]
        roots: list[tuple[float, int]] = []
        for idx in range(len(tau_grid) - 1):
            f_lo = residuals[idx]
            f_hi = residuals[idx + 1]
            if abs(f_lo) <= self.config.root_tolerance:
                roots.append((float(tau_grid[idx]), 0))
                continue
            if f_lo * f_hi > 0.0:
                continue
            roots.append(
                self._bisection(
                    theta,
                    float(tau_grid[idx]),
                    float(tau_grid[idx + 1]),
                    f_corona=f_corona,
                    feedback_factor=feedback_factor,
                )
            )

        deduped: list[tuple[float, int]] = []
        for root, iterations in roots:
            if not deduped or abs(math.log(root / deduped[-1][0])) > 1.0e-6:
                deduped.append((root, iterations))
        return deduped

    def find_tau_root(
        self,
        theta: float,
        *,
        f_corona: float,
        feedback_factor: float,
        guess_tau: float | None,
    ) -> tuple[float, str, int]:
        if guess_tau is not None:
            guess_tau = min(max(guess_tau, self.config.tau_min), self.config.tau_max)
            guess_residual = self.energy_residual(theta, guess_tau, f_corona=f_corona, feedback_factor=feedback_factor)
            if abs(guess_residual) <= self.config.root_tolerance:
                return guess_tau, "continuation", 0

            lower = guess_tau
            upper = guess_tau
            lower_residual = guess_residual
            upper_residual = guess_residual
            for _ in range(self.config.continuation_expand_steps):
                if lower > self.config.tau_min:
                    lower = max(lower / self.config.continuation_expand_factor, self.config.tau_min)
                    lower_residual = self.energy_residual(
                        theta,
                        lower,
                        f_corona=f_corona,
                        feedback_factor=feedback_factor,
                    )
                if lower_residual * guess_residual <= 0.0:
                    root, iterations = self._bisection(
                        theta,
                        lower,
                        guess_tau,
                        f_corona=f_corona,
                        feedback_factor=feedback_factor,
                    )
                    return root, "continuation", iterations

                if upper < self.config.tau_max:
                    upper = min(upper * self.config.continuation_expand_factor, self.config.tau_max)
                    upper_residual = self.energy_residual(
                        theta,
                        upper,
                        f_corona=f_corona,
                        feedback_factor=feedback_factor,
                    )
                if upper_residual * guess_residual <= 0.0:
                    root, iterations = self._bisection(
                        theta,
                        guess_tau,
                        upper,
                        f_corona=f_corona,
                        feedback_factor=feedback_factor,
                    )
                    return root, "continuation", iterations

        roots = self._global_roots(theta, f_corona=f_corona, feedback_factor=feedback_factor)
        if not roots:
            raise RuntimeError(
                "No f-g energy-balance root found for "
                f"kTe={theta * MEC2_KEV:.6g} keV, f={f_corona:g}, g={feedback_factor:g}."
            )
        if guess_tau is None:
            root, iterations = max(roots, key=lambda item: item[0])
            return root, "global", iterations
        root, iterations = min(roots, key=lambda item: abs(math.log(item[0] / guess_tau)))
        return root, "global", iterations

    def solve_point(
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
        row = self.evaluate_tau(theta, tau_t, f_corona=f_corona, feedback_factor=feedback_factor)
        row["root_method"] = root_method
        row["root_iterations"] = root_iterations
        return row


def scan_fg_curves(config: FGEnergyBalanceConfig) -> list[dict[str, float | int | bool | str]]:
    solver = FGEnergyBalanceSolver(config)
    theta_values = [
        value / MEC2_KEV
        for value in logspace(config.kTe_min_kev, config.kTe_max_kev, config.n_samples)
    ]
    rows: list[dict[str, float | int | bool | str]] = []
    for f_corona in config.f_values:
        for feedback_factor in config.g_values:
            previous_tau: float | None = None
            for theta in theta_values:
                row = solver.solve_point(
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
                    f"Areq={float(row['A_required']):.6g} a={float(row['effective_albedo']):.4g} "
                    f"res={float(row['energy_log_residual']):.3g}",
                    flush=True,
                )
    return rows


def write_rows(path: pathlib.Path, rows: Sequence[dict[str, float | int | bool | str]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty f-g scan")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(
    path: pathlib.Path,
    rows: Sequence[dict[str, float | int | bool | str]],
    *,
    f_values: Sequence[float],
    g_values: Sequence[float],
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_comppsc_fg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.9, len(f_values)))
    linestyles = ["-", "--", ":", "-."]
    for f_idx, f_corona in enumerate(f_values):
        for g_idx, feedback_factor in enumerate(g_values):
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
            ax.plot(
                [float(row["kTe_keV"]) for row in selected],
                [float(row["tau_T"]) for row in selected],
                color=colors[f_idx],
                linestyle=linestyles[g_idx % len(linestyles)],
                lw=2.0,
                marker="o",
                markersize=3.2,
                label=rf"$f={f_corona:g},\,g={feedback_factor:g}$",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$kT_{\rm e}$ (keV)")
    ax.set_ylabel(r"$\tau_{\rm T}$")
    ax.set_title(r"compPSc + ireflect equilibrium curves")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(frameon=False, fontsize=8.2, ncols=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kTe-min", type=float, default=10.0)
    parser.add_argument("--kTe-max", type=float, default=200.0)
    parser.add_argument("--n-samples", type=int, default=24)
    parser.add_argument("--f-values", nargs="*", default=None, help="Space/comma separated coronal fractions.")
    parser.add_argument("--g-values", nargs="*", default=None, help="Space/comma separated feedback factors.")
    parser.add_argument("--fixed-xi", type=float, default=100.0)
    parser.add_argument("--max-scatter", type=int, default=2000)
    parser.add_argument("--mu-order", type=int, default=8)
    parser.add_argument("--tau-min", type=float, default=0.03)
    parser.add_argument("--tau-max", type=float, default=10.0)
    parser.add_argument("--root-tolerance", type=float, default=5.0e-4)
    parser.add_argument("--global-tau-samples", type=int, default=18)
    parser.add_argument("--output-csv", type=pathlib.Path, default=None)
    parser.add_argument("--output-png", type=pathlib.Path, default=None)
    parser.add_argument("--no-exact-angles", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    f_values = parse_float_list(args.f_values, FGEnergyBalanceConfig.f_values)
    g_values = parse_float_list(args.g_values, FGEnergyBalanceConfig.g_values)
    config = FGEnergyBalanceConfig(
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
    rows = scan_fg_curves(config)
    default_csv, default_png = output_paths(f_values, g_values, args.fixed_xi)
    output_csv = args.output_csv or default_csv
    output_png = args.output_png or default_png
    write_rows(output_csv, rows)
    plot_curves(output_png, rows, f_values=f_values, g_values=g_values)
    print(output_csv)
    print(output_png)


if __name__ == "__main__":
    main()
