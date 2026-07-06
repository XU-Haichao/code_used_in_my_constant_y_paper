#!/usr/bin/env python3
"""Self-consistent compPSc/reflection feedback for a slab corona."""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np


ROOT_FOR_IMPORT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))


@dataclass(frozen=True)
class DiskResponse:
    returned_seed_bins: np.ndarray
    downward_energy: float
    reflected_energy: float
    absorbed_energy: float
    energy_residual: float


@dataclass(frozen=True)
class FeedbackGeneration:
    returned_seed_bins: np.ndarray
    input_energy: float
    unscattered_up_energy: float
    compton_up_energy: float
    compton_down_energy: float
    reflected_energy: float
    absorbed_energy: float
    coronal_gain_energy: float
    disk_energy_residual: float
    last_scatter_order: int
    last_difmax: float


@dataclass(frozen=True)
class EigenmodeResult:
    lambda_energy: float
    shape_residual: float
    iterations: int
    seed_shape: np.ndarray
    generation: FeedbackGeneration
    converged: bool


@dataclass(frozen=True)
class ImpulseResult:
    history: tuple[dict[str, float | int], ...]
    total_input_energy: float
    total_blackbody_emitted_energy: float
    total_comptonized_energy: float
    total_upward_escape_energy: float
    total_reflected_energy: float
    total_absorbed_energy: float
    total_coronal_gain_energy: float
    remaining_feedback_energy: float
    converged: bool


@dataclass(frozen=True)
class FullFeedbackConfig:
    kTe_kev: float = 10.0
    tbb_kev: float = 0.005
    max_scatter: int = 2000
    exact_angles: bool = True
    observer_mu: float = 0.5
    hemisphere_mu_order: int = 12
    reflector_abundance: float = 1.0
    reflector_iron_abundance: float = 1.0


class FullFeedbackComppscSolver:
    def __init__(self, config: FullFeedbackConfig):
        from pair_balance.scanner_comppsc import ComppscScanConfig, ComppscSlabSolver
        from pair_balance.scanner_reflect import NeutralReflectionKernel, ReflectionConfig

        self.config = config
        theta = config.kTe_kev / 511.0
        transfer_config = ComppscScanConfig(
            theta_min=theta,
            theta_max=theta,
            n_samples=1,
            tbb_kev=config.tbb_kev,
            max_scatter=config.max_scatter,
            exact_angles=config.exact_angles,
            observer_mu=config.observer_mu,
        )
        self.transfer = ComppscSlabSolver(transfer_config)
        reflection_config = ReflectionConfig(
            hemisphere_mu_order=config.hemisphere_mu_order,
            reflector_abundance=config.reflector_abundance,
            reflector_iron_abundance=config.reflector_iron_abundance,
        )
        self.reflector = NeutralReflectionKernel(reflection_config)

        bootstrap = self.transfer.run_state(theta, 1.0)
        self.x_grid = np.asarray(bootstrap.x_grid, dtype=float)
        self.x_weights = np.asarray(bootstrap.x_weights, dtype=float)
        self.energy_edges_kev = native_energy_edges(self.x_grid)
        self.energy_weights = seed_energy_weights(self.energy_edges_kev)
        self.initial_blackbody_bins = blackbody_seed_bins(
            self.energy_edges_kev,
            tbb_kev=config.tbb_kev,
            target_energy=1.0,
        )

    def _native_transfer_spectra(
        self,
        seed_bins: np.ndarray,
        tau_t: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        requested_input_energy = float(np.dot(self.energy_weights, seed_bins))
        if requested_input_energy <= 0.0:
            raise ValueError("seed spectrum must carry positive energy")

        self.transfer.seed_edges = np.asarray(self.energy_edges_kev, dtype=np.float64)
        self.transfer.seed_spec = np.asarray(seed_bins, dtype=np.float64)
        self.transfer._configure_transfer()
        parm = np.array(
            [
                self.config.kTe_kev,
                2.0,
                -1.0,
                1000.0,
                tau_t,
                1.0,
                1.0,
                self.config.observer_mu,
                1.0,
                0.0,
            ],
            dtype=np.float64,
        )
        self.transfer._run_msismco(parm)

        module = self.transfer.module
        mu = np.asarray(module.msqqcm2.uang, dtype=float)
        mu_weights = np.asarray(module.msqqcm2.aang, dtype=float)
        unscattered_up = self.transfer._angular_flux(module.msqqres.dintpl, mu, mu_weights)
        compton_up = self.transfer._angular_flux(module.msqqres.suipl, mu, mu_weights)
        compton_down = self.transfer._angular_flux(module.msqqres.suimi, mu, mu_weights)
        model_input_energy = energy_flux(
            self.x_grid,
            self.x_weights,
            np.asarray(module.msqbb.dinten, dtype=float),
        )
        if model_input_energy <= 0.0:
            raise RuntimeError("compPSc mapped the seed spectrum to zero energy")

        remap_correction = requested_input_energy / model_input_energy
        return (
            np.maximum(unscattered_up * remap_correction, 0.0),
            np.maximum(compton_up * remap_correction, 0.0),
            np.maximum(compton_down * remap_correction, 0.0),
            requested_input_energy,
        )

    def round_trip(self, seed_bins: np.ndarray, tau_t: float) -> FeedbackGeneration:
        unscattered_up, compton_up, compton_down, input_energy = self._native_transfer_spectra(seed_bins, tau_t)
        _, reflected, _, _ = self.reflector.hemisphere_response(
            self.x_grid,
            self.x_weights,
            compton_down,
            self.config.observer_mu,
        )
        response = disk_response(
            self.x_grid,
            self.x_weights,
            self.energy_edges_kev,
            compton_down,
            reflected,
            self.initial_blackbody_bins,
        )

        unscattered_up_energy = energy_flux(self.x_grid, self.x_weights, unscattered_up)
        compton_up_energy = energy_flux(self.x_grid, self.x_weights, compton_up)
        compton_down_energy = response.downward_energy
        output_energy = unscattered_up_energy + compton_up_energy + compton_down_energy
        module = self.transfer.module
        last_scatter = int(getattr(module.msiterstat, "lastisc", -1))
        last_difmax = float(getattr(module.msiterstat, "lastdifmax", math.nan))
        return FeedbackGeneration(
            returned_seed_bins=response.returned_seed_bins,
            input_energy=input_energy,
            unscattered_up_energy=unscattered_up_energy,
            compton_up_energy=compton_up_energy,
            compton_down_energy=compton_down_energy,
            reflected_energy=response.reflected_energy,
            absorbed_energy=response.absorbed_energy,
            coronal_gain_energy=output_energy - input_energy,
            disk_energy_residual=response.energy_residual,
            last_scatter_order=last_scatter,
            last_difmax=last_difmax,
        )


def energy_flux(x_grid: np.ndarray, x_weights: np.ndarray, spectrum: np.ndarray) -> float:
    x = np.asarray(x_grid, dtype=float)
    weights = np.asarray(x_weights, dtype=float)
    values = np.asarray(spectrum, dtype=float)
    if x.shape != weights.shape or x.shape != values.shape:
        raise ValueError("x_grid, x_weights, and spectrum must have identical shapes")
    return float(np.sum(x * weights * values))


def seed_energy_weights(energy_edges_kev: np.ndarray) -> np.ndarray:
    edges = np.asarray(energy_edges_kev, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("energy edges must be a strictly increasing one-dimensional array")
    return np.sqrt(edges[:-1] * edges[1:]) / 511.0


def native_energy_edges(x_grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x_grid, dtype=float)
    if x.ndim != 1 or x.size < 2 or np.any(x <= 0.0) or np.any(np.diff(x) <= 0.0):
        raise ValueError("x_grid must be a strictly increasing positive array")
    log_centers = np.log(511.0 * x)
    log_edges = np.empty(x.size + 1, dtype=float)
    log_edges[1:-1] = 0.5 * (log_centers[:-1] + log_centers[1:])
    log_edges[0] = 2.0 * log_centers[0] - log_edges[1]
    log_edges[-1] = 2.0 * log_centers[-1] - log_edges[-2]
    return np.exp(log_edges)


def blackbody_seed_bins(
    energy_edges_kev: np.ndarray,
    *,
    tbb_kev: float,
    target_energy: float = 1.0,
) -> np.ndarray:
    edges = np.asarray(energy_edges_kev, dtype=float)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = np.diff(edges)
    scaled_energy = centers / float(tbb_kev)
    bins = np.zeros_like(centers)
    rayleigh_jeans = scaled_energy < 1.0e-8
    finite = (scaled_energy >= 1.0e-8) & (scaled_energy < 700.0)
    bins[rayleigh_jeans] = centers[rayleigh_jeans] * tbb_kev * widths[rayleigh_jeans]
    bins[finite] = centers[finite] ** 2 / np.expm1(scaled_energy[finite]) * widths[finite]
    return normalize_seed_energy(edges, bins, target_energy=target_energy)


def normalize_seed_energy(
    energy_edges_kev: np.ndarray,
    seed_bins: np.ndarray,
    *,
    target_energy: float = 1.0,
) -> np.ndarray:
    bins = np.maximum(np.asarray(seed_bins, dtype=float), 0.0)
    weights = seed_energy_weights(energy_edges_kev)
    if bins.shape != weights.shape:
        raise ValueError("seed bins must have one entry per energy bin")
    current = float(np.dot(weights, bins))
    if current <= 0.0:
        raise ValueError("cannot normalize a seed spectrum with zero energy")
    return bins * (float(target_energy) / current)


def spectrum_to_seed_bins(
    x_grid: np.ndarray,
    x_weights: np.ndarray,
    energy_edges_kev: np.ndarray,
    spectrum: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x_grid, dtype=float)
    edges = np.asarray(energy_edges_kev, dtype=float)
    values = np.maximum(np.asarray(spectrum, dtype=float), 0.0)
    if edges.size != x.size + 1 or values.shape != x.shape:
        raise ValueError("native energy edges and spectrum must match x_grid")

    centers_kev = 511.0 * x
    raw_bins = values * np.diff(edges) / centers_kev
    target_energy = energy_flux(x, np.asarray(x_weights, dtype=float), values)
    if target_energy <= 0.0:
        return np.zeros_like(values)
    return normalize_seed_energy(edges, raw_bins, target_energy=target_energy)


def disk_response(
    x_grid: np.ndarray,
    x_weights: np.ndarray,
    energy_edges_kev: np.ndarray,
    downward_spectrum: np.ndarray,
    reflected_spectrum: np.ndarray,
    unit_blackbody_bins: np.ndarray,
    *,
    reflection_tolerance: float = 1.0e-8,
) -> DiskResponse:
    downward = np.maximum(np.asarray(downward_spectrum, dtype=float), 0.0)
    reflected = np.maximum(np.asarray(reflected_spectrum, dtype=float), 0.0)
    downward_energy = energy_flux(x_grid, x_weights, downward)
    reflected_energy = energy_flux(x_grid, x_weights, reflected)

    if reflected_energy > downward_energy * (1.0 + reflection_tolerance):
        raise ValueError(
            "reflected energy exceeds downward incident energy: "
            f"{reflected_energy:.8g} > {downward_energy:.8g}"
        )
    if reflected_energy > downward_energy and reflected_energy > 0.0:
        reflected *= downward_energy / reflected_energy
        reflected_energy = downward_energy

    absorbed_energy = max(downward_energy - reflected_energy, 0.0)
    reflected_bins = spectrum_to_seed_bins(
        x_grid,
        x_weights,
        energy_edges_kev,
        reflected,
    )
    blackbody_bins = normalize_seed_energy(
        energy_edges_kev,
        unit_blackbody_bins,
        target_energy=absorbed_energy,
    ) if absorbed_energy > 0.0 else np.zeros_like(unit_blackbody_bins, dtype=float)
    returned = reflected_bins + blackbody_bins
    returned_energy = float(np.dot(seed_energy_weights(energy_edges_kev), returned))
    residual = returned_energy - downward_energy
    return DiskResponse(
        returned_seed_bins=returned,
        downward_energy=downward_energy,
        reflected_energy=reflected_energy,
        absorbed_energy=absorbed_energy,
        energy_residual=residual,
    )


def power_iteration(
    round_trip: Callable[[np.ndarray, float], FeedbackGeneration],
    *,
    tau_t: float,
    initial_seed_bins: np.ndarray,
    energy_weights: np.ndarray,
    tolerance: float = 1.0e-6,
    max_iterations: int = 80,
) -> EigenmodeResult:
    weights = np.asarray(energy_weights, dtype=float)
    seed = np.maximum(np.asarray(initial_seed_bins, dtype=float), 0.0)
    initial_energy = float(np.dot(weights, seed))
    if initial_energy <= 0.0:
        raise ValueError("initial seed must carry positive energy")
    seed = seed / initial_energy

    last_lambda = math.nan
    last_residual = math.inf
    last_generation: FeedbackGeneration | None = None
    for iteration in range(1, max_iterations + 1):
        generation = round_trip(seed, tau_t)
        returned_energy = float(np.dot(weights, generation.returned_seed_bins))
        if returned_energy <= 0.0:
            raise RuntimeError("round-trip operator returned zero energy")
        new_seed = generation.returned_seed_bins / returned_energy
        shape_residual = float(np.sum(np.abs(weights * (new_seed - seed))))
        lambda_residual = abs(returned_energy - last_lambda) if math.isfinite(last_lambda) else math.inf
        seed = new_seed
        last_lambda = returned_energy
        last_residual = shape_residual
        last_generation = generation
        if shape_residual <= tolerance and lambda_residual <= tolerance:
            return EigenmodeResult(
                lambda_energy=returned_energy,
                shape_residual=shape_residual,
                iterations=iteration,
                seed_shape=seed,
                generation=generation,
                converged=True,
            )

    assert last_generation is not None
    return EigenmodeResult(
        lambda_energy=last_lambda,
        shape_residual=last_residual,
        iterations=max_iterations,
        seed_shape=seed,
        generation=last_generation,
        converged=False,
    )


def bisect_feedback_root(
    evaluator: Callable[[float], float],
    *,
    tau_lo: float,
    tau_hi: float,
    relative_tolerance: float = 1.0e-5,
    max_iterations: int = 60,
) -> tuple[float, float, int]:
    if not (0.0 < tau_lo < tau_hi):
        raise ValueError("tau bracket must satisfy 0 < tau_lo < tau_hi")
    f_lo = math.log(float(evaluator(tau_lo)))
    f_hi = math.log(float(evaluator(tau_hi)))
    if f_lo * f_hi > 0.0:
        raise ValueError("feedback root is not bracketed")

    for iteration in range(1, max_iterations + 1):
        tau_mid = math.sqrt(tau_lo * tau_hi)
        f_mid = math.log(float(evaluator(tau_mid)))
        if abs(f_mid) <= relative_tolerance:
            return tau_mid, math.expm1(f_mid), iteration
        if f_lo * f_mid <= 0.0:
            tau_hi = tau_mid
            f_hi = f_mid
        else:
            tau_lo = tau_mid
            f_lo = f_mid

    tau_mid = math.sqrt(tau_lo * tau_hi)
    f_mid = math.log(float(evaluator(tau_mid)))
    return tau_mid, math.expm1(f_mid), max_iterations


def run_impulse_response(
    round_trip: Callable[[np.ndarray, float], FeedbackGeneration],
    *,
    tau_t: float,
    initial_seed_bins: np.ndarray,
    energy_weights: np.ndarray,
    stop_energy: float = 1.0e-8,
    max_generations: int = 200,
) -> ImpulseResult:
    weights = np.asarray(energy_weights, dtype=float)
    seed = np.maximum(np.asarray(initial_seed_bins, dtype=float), 0.0)
    initial_energy = float(np.dot(weights, seed))
    if initial_energy <= 0.0:
        raise ValueError("initial seed must carry positive energy")

    history: list[dict[str, float | int]] = []
    total_input = 0.0
    total_comptonized = 0.0
    total_upward = 0.0
    total_reflected = 0.0
    total_absorbed = 0.0
    total_gain = 0.0

    for generation_index in range(1, max_generations + 1):
        input_energy = float(np.dot(weights, seed))
        if input_energy <= stop_energy * initial_energy:
            break
        generation = round_trip(seed, tau_t)
        returned_energy = float(np.dot(weights, generation.returned_seed_bins))
        upward_energy = generation.unscattered_up_energy + generation.compton_up_energy
        comptonized_energy = generation.compton_up_energy + generation.compton_down_energy
        history.append(
            {
                "generation": generation_index,
                "input_energy": input_energy,
                "unscattered_up_energy": generation.unscattered_up_energy,
                "compton_up_energy": generation.compton_up_energy,
                "compton_down_energy": generation.compton_down_energy,
                "upward_escape_energy": upward_energy,
                "reflected_energy": generation.reflected_energy,
                "absorbed_blackbody_energy": generation.absorbed_energy,
                "returned_feedback_energy": returned_energy,
                "coronal_gain_energy": generation.coronal_gain_energy,
                "disk_energy_residual": generation.disk_energy_residual,
                "last_scatter_order": generation.last_scatter_order,
                "last_difmax": generation.last_difmax,
            }
        )
        total_input += input_energy
        total_comptonized += comptonized_energy
        total_upward += upward_energy
        total_reflected += generation.reflected_energy
        total_absorbed += generation.absorbed_energy
        total_gain += generation.coronal_gain_energy
        seed = generation.returned_seed_bins

    remaining = float(np.dot(weights, seed))
    return ImpulseResult(
        history=tuple(history),
        total_input_energy=total_input,
        total_blackbody_emitted_energy=initial_energy + total_absorbed,
        total_comptonized_energy=total_comptonized,
        total_upward_escape_energy=total_upward,
        total_reflected_energy=total_reflected,
        total_absorbed_energy=total_absorbed,
        total_coronal_gain_energy=total_gain,
        remaining_feedback_energy=remaining,
        converged=remaining <= stop_energy * initial_energy,
    )


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pair_balance" / "data"
OUTPUT_DIR = ROOT / "output"
SUMMARY_CSV = DATA_DIR / "comppsc_full_feedback_10kev_summary.csv"
SCAN_CSV = DATA_DIR / "comppsc_full_feedback_10kev_tau_scan.csv"
ITERATIONS_CSV = DATA_DIR / "comppsc_full_feedback_10kev_iterations.csv"
COMPARISON_PNG = OUTPUT_DIR / "comppsc_full_feedback_10kev_comparison.png"


def _write_rows(path: pathlib.Path, rows: list[dict[str, float | int | bool | str]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _find_unit_bracket(samples: list[tuple[float, float]]) -> tuple[float, float]:
    ordered = sorted(samples)
    for (tau_lo, value_lo), (tau_hi, value_hi) in zip(ordered[:-1], ordered[1:]):
        if (value_lo - 1.0) * (value_hi - 1.0) <= 0.0:
            return tau_lo, tau_hi
    raise RuntimeError("the sampled optical-depth range does not bracket lambda=1")


def _plot_analysis(
    scan_rows: list[dict[str, float | int | bool | str]],
    impulse: ImpulseResult,
    *,
    tau_fixed: float,
    tau_one_pass: float,
    tau_full: float,
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_comppsc_full_feedback")
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax_gain, ax_history) = plt.subplots(1, 2, figsize=(13.2, 5.2))

    ordered = sorted(scan_rows, key=lambda row: float(row["tau_T"]))
    ax_gain.plot(
        [float(row["tau_T"]) for row in ordered],
        [float(row["lambda_full_feedback"]) for row in ordered],
        color="#0072B2",
        marker="o",
        markersize=3.5,
        lw=1.8,
        label="Full spectral feedback",
    )
    ax_gain.axhline(1.0, color="#222222", lw=1.2)
    ax_gain.axvline(tau_fixed, color="#D55E00", ls="--", lw=1.8, label="Fixed albedo a=0.2")
    ax_gain.axvline(tau_one_pass, color="#009E73", ls=":", lw=2.2, label="One-pass reflect")
    ax_gain.axvline(tau_full, color="#7C3AED", ls="-", lw=1.8, label="Full-feedback root")
    ax_gain.set_xlabel(r"$\tau_{\rm T}$")
    ax_gain.set_ylabel(r"Round-trip energy gain $\lambda$")
    ax_gain.set_title(r"Equilibrium root at $kT_{\rm e}=10$ keV")
    ax_gain.legend()

    history = list(impulse.history)
    generation = [int(row["generation"]) for row in history]
    history_series = (
        ("input_energy", "Input to corona", "#0072B2"),
        ("returned_feedback_energy", "Returned feedback", "#7C3AED"),
        ("upward_escape_energy", "Upward escape", "#D55E00"),
        ("absorbed_blackbody_energy", "5 eV reprocessing", "#009E73"),
        ("reflected_energy", "Reflected reinjection", "#CC79A7"),
    )
    for key, label, color in history_series:
        ax_history.plot(
            generation,
            [max(float(row[key]), 1.0e-12) for row in history],
            lw=1.9,
            color=color,
            label=label,
        )
    ax_history.set_yscale("log")
    ax_history.set_xlabel("Feedback generation")
    ax_history.set_ylabel("Energy relative to initial 5 eV input")
    ax_history.set_title("Impulse history evaluated at the equilibrium root")
    ax_history.legend()

    fig.tight_layout()
    fig.savefig(COMPARISON_PNG, dpi=200)
    plt.close(fig)


def run_full_analysis(
    config: FullFeedbackConfig,
    *,
    tau_min: float = 1.0,
    tau_max: float = 4.0,
    tau_samples: int = 9,
    shape_tolerance: float = 3.0e-5,
    root_tolerance: float = 2.0e-4,
    impulse_generations: int = 40,
) -> dict[str, float | int | bool | str]:
    solver = FullFeedbackComppscSolver(config)
    full_cache: dict[float, EigenmodeResult] = {}
    generation_cache: dict[float, FeedbackGeneration] = {}

    def cache_key(tau_t: float) -> float:
        return round(float(tau_t), 10)

    def full_result(tau_t: float) -> EigenmodeResult:
        key = cache_key(tau_t)
        if key not in full_cache:
            result = power_iteration(
                solver.round_trip,
                tau_t=tau_t,
                initial_seed_bins=solver.initial_blackbody_bins,
                energy_weights=solver.energy_weights,
                tolerance=shape_tolerance,
                max_iterations=60,
            )
            if not result.converged:
                raise RuntimeError(f"feedback spectrum did not converge at tau={tau_t:.6g}")
            full_cache[key] = result
            generation = result.generation
            print(
                f"tau={tau_t:.6f} lambda={result.lambda_energy:.7f} "
                f"shape_n={result.iterations} sc={generation.last_scatter_order} "
                f"dif={generation.last_difmax:.3g}",
                flush=True,
            )
        return full_cache[key]

    def initial_generation(tau_t: float) -> FeedbackGeneration:
        key = cache_key(tau_t)
        if key not in generation_cache:
            generation_cache[key] = solver.round_trip(solver.initial_blackbody_bins, tau_t)
        return generation_cache[key]

    sampled_tau = np.geomspace(tau_min, tau_max, tau_samples)
    sampled_full = [(float(tau_t), full_result(float(tau_t)).lambda_energy) for tau_t in sampled_tau]
    full_lo, full_hi = _find_unit_bracket(sampled_full)
    tau_full, full_residual, full_root_iterations = bisect_feedback_root(
        lambda tau_t: full_result(tau_t).lambda_energy,
        tau_lo=full_lo,
        tau_hi=full_hi,
        relative_tolerance=root_tolerance,
    )
    root_mode = full_result(tau_full)

    fixed_samples = [
        (float(tau_t), 0.8 * initial_generation(float(tau_t)).compton_down_energy)
        for tau_t in sampled_tau
    ]
    fixed_lo, fixed_hi = _find_unit_bracket(fixed_samples)
    tau_fixed, fixed_residual, fixed_root_iterations = bisect_feedback_root(
        lambda tau_t: 0.8 * initial_generation(tau_t).compton_down_energy,
        tau_lo=fixed_lo,
        tau_hi=fixed_hi,
        relative_tolerance=root_tolerance,
    )

    one_pass_samples = [
        (float(tau_t), initial_generation(float(tau_t)).absorbed_energy)
        for tau_t in sampled_tau
    ]
    one_lo, one_hi = _find_unit_bracket(one_pass_samples)
    tau_one_pass, one_residual, one_root_iterations = bisect_feedback_root(
        lambda tau_t: initial_generation(tau_t).absorbed_energy,
        tau_lo=one_lo,
        tau_hi=one_hi,
        relative_tolerance=root_tolerance,
    )

    impulse = run_impulse_response(
        solver.round_trip,
        tau_t=tau_full,
        initial_seed_bins=solver.initial_blackbody_bins,
        energy_weights=solver.energy_weights,
        stop_energy=1.0e-8,
        max_generations=impulse_generations,
    )
    root_generation = root_mode.generation
    per_cycle_comptonized = root_generation.compton_up_energy + root_generation.compton_down_energy
    per_cycle_upward = root_generation.unscattered_up_energy + root_generation.compton_up_energy
    effective_albedo = root_generation.reflected_energy / max(root_generation.compton_down_energy, 1.0e-30)
    initial_mean_photon_energy_kev = 511.0 / float(np.sum(solver.initial_blackbody_bins))

    summary: dict[str, float | int | bool | str] = {
        "model": "compPSc+reflect full feedback",
        "kTe_keV": config.kTe_kev,
        "theta": config.kTe_kev / 511.0,
        "tbb_keV": config.tbb_kev,
        "initial_blackbody_mean_photon_energy_keV": initial_mean_photon_energy_kev,
        "tau_fixed_albedo_0p2": tau_fixed,
        "tau_one_pass_reflect": tau_one_pass,
        "tau_full_feedback": tau_full,
        "tau_full_minus_fixed": tau_full - tau_fixed,
        "tau_full_over_fixed": tau_full / tau_fixed,
        "tau_full_minus_one_pass": tau_full - tau_one_pass,
        "lambda_full_feedback": root_mode.lambda_energy,
        "lambda_root_residual": full_residual,
        "shape_iterations": root_mode.iterations,
        "shape_residual": root_mode.shape_residual,
        "fixed_root_residual": fixed_residual,
        "one_pass_root_residual": one_residual,
        "full_root_iterations": full_root_iterations,
        "fixed_root_iterations": fixed_root_iterations,
        "one_pass_root_iterations": one_root_iterations,
        "per_cycle_input_energy": root_generation.input_energy,
        "per_cycle_blackbody_5eV_energy": root_generation.absorbed_energy,
        "per_cycle_reflected_reinjection_energy": root_generation.reflected_energy,
        "per_cycle_comptonized_energy": per_cycle_comptonized,
        "per_cycle_unscattered_up_energy": root_generation.unscattered_up_energy,
        "per_cycle_upward_escape_energy": per_cycle_upward,
        "per_cycle_coronal_gain_energy": root_generation.coronal_gain_energy,
        "per_initial_photon_cycle_blackbody_5eV_energy_keV": (
            root_generation.absorbed_energy * initial_mean_photon_energy_kev
        ),
        "per_initial_photon_cycle_comptonized_energy_keV": (
            per_cycle_comptonized * initial_mean_photon_energy_kev
        ),
        "per_initial_photon_cycle_upward_escape_energy_keV": (
            per_cycle_upward * initial_mean_photon_energy_kev
        ),
        "effective_albedo": effective_albedo,
        "last_scatter_order": root_generation.last_scatter_order,
        "last_difmax": root_generation.last_difmax,
        "disk_energy_residual": root_generation.disk_energy_residual,
        "impulse_generations": len(impulse.history),
        "impulse_converged": impulse.converged,
        "impulse_total_blackbody_5eV_energy": impulse.total_blackbody_emitted_energy,
        "impulse_total_comptonized_energy": impulse.total_comptonized_energy,
        "impulse_total_upward_escape_energy": impulse.total_upward_escape_energy,
        "impulse_remaining_feedback_energy": impulse.remaining_feedback_energy,
        "impulse_total_blackbody_5eV_energy_per_initial_photon_keV": (
            impulse.total_blackbody_emitted_energy * initial_mean_photon_energy_kev
        ),
        "impulse_total_comptonized_energy_per_initial_photon_keV": (
            impulse.total_comptonized_energy * initial_mean_photon_energy_kev
        ),
        "impulse_note": "At lambda=1 the energy impulse does not decay; cumulative totals depend on max_generations.",
        "max_scatter": config.max_scatter,
        "exact_angles": config.exact_angles,
        "hemisphere_mu_order": config.hemisphere_mu_order,
    }

    scan_rows: list[dict[str, float | int | bool | str]] = []
    for tau_key, result in sorted(full_cache.items()):
        generation = result.generation
        scan_rows.append(
            {
                "tau_T": tau_key,
                "lambda_full_feedback": result.lambda_energy,
                "shape_iterations": result.iterations,
                "shape_residual": result.shape_residual,
                "input_energy": generation.input_energy,
                "unscattered_up_energy": generation.unscattered_up_energy,
                "compton_up_energy": generation.compton_up_energy,
                "compton_down_energy": generation.compton_down_energy,
                "reflected_energy": generation.reflected_energy,
                "absorbed_energy": generation.absorbed_energy,
                "coronal_gain_energy": generation.coronal_gain_energy,
                "last_scatter_order": generation.last_scatter_order,
                "last_difmax": generation.last_difmax,
            }
        )
    impulse_rows = [dict(row, tau_T=tau_full) for row in impulse.history]
    _write_rows(SUMMARY_CSV, [summary])
    _write_rows(SCAN_CSV, scan_rows)
    _write_rows(ITERATIONS_CSV, impulse_rows)
    _plot_analysis(
        scan_rows,
        impulse,
        tau_fixed=tau_fixed,
        tau_one_pass=tau_one_pass,
        tau_full=tau_full,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("point", "analyze"), default="analyze")
    parser.add_argument("--kTe", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--tau-min", type=float, default=1.0)
    parser.add_argument("--tau-max", type=float, default=4.0)
    parser.add_argument("--tau-samples", type=int, default=9)
    parser.add_argument("--max-scatter", type=int, default=2000)
    parser.add_argument("--mu-order", type=int, default=12)
    parser.add_argument("--impulse-generations", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FullFeedbackConfig(
        kTe_kev=args.kTe,
        max_scatter=args.max_scatter,
        hemisphere_mu_order=args.mu_order,
    )
    if args.mode == "point":
        solver = FullFeedbackComppscSolver(config)
        generation = solver.round_trip(solver.initial_blackbody_bins, args.tau)
        returned = float(np.dot(solver.energy_weights, generation.returned_seed_bins))
        print(
            f"kTe={args.kTe:g} tau={args.tau:g} input={generation.input_energy:.8g} "
            f"down={generation.compton_down_energy:.8g} returned={returned:.8g} "
            f"reflected={generation.reflected_energy:.8g} absorbed={generation.absorbed_energy:.8g} "
            f"disk_residual={generation.disk_energy_residual:.3g} "
            f"sc={generation.last_scatter_order} dif={generation.last_difmax:.3g}"
        )
        return

    summary = run_full_analysis(
        config,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        tau_samples=args.tau_samples,
        impulse_generations=args.impulse_generations,
    )
    for key, value in summary.items():
        print(f"{key}={value}")
    print(SUMMARY_CSV)
    print(SCAN_CSV)
    print(ITERATIONS_CSV)
    print(COMPARISON_PNG)


if __name__ == "__main__":
    main()
