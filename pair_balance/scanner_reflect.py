#!/usr/bin/env python3
"""Reflection-coupled slab pair-line solver.

This module keeps the fixed-albedo slab closure from ``scanner.py`` and adds a
second branch in which the neutral reflection hump is computed from the actual
downward COMPPS illumination using the HEASoft/XSPEC ``reflect`` model
(Magdziarz & Zdziarski 1995).

The reflection branch does two things beyond the fixed-albedo solver:

1. For each slab state, compute the angle-dependent reflected hump from the
   downward illuminating spectrum returned by COMPPS.
2. Replace the constant cold-disk albedo by an energy- and spectrum-dependent
   effective albedo obtained by integrating the reflected luminosity over the
   upper hemisphere.

The remaining limitation is unchanged from the earlier discussion: stock COMPPS
cannot accept an arbitrary reflected bottom boundary source, so the reflected
hump is used in the disk energy partition and in the observable spectrum, but
is not re-transported through the corona as a separate boundary source.
"""

from __future__ import annotations

import csv
import ctypes
import math
import os
import pathlib
import subprocess
from dataclasses import dataclass

import numpy as np

try:
    from pair_balance.scanner import (
        CLIGHT,
        KTE_TO_THETA,
        MEC2_ERG,
        OUTPUT_DIR,
        R_E,
        SIGMA_T,
        SLAB_HEIGHT_CM,
        ComppsSlabSolver,
        RadiativeState,
        ScanConfig,
        logspace,
    )
except ModuleNotFoundError:
    from scanner import (
        CLIGHT,
        KTE_TO_THETA,
        MEC2_ERG,
        OUTPUT_DIR,
        R_E,
        SIGMA_T,
        SLAB_HEIGHT_CM,
        ComppsSlabSolver,
        RadiativeState,
        ScanConfig,
        logspace,
    )


BASE = pathlib.Path(__file__).resolve().parent
BUILD_DIR = BASE / "_build"
WRAPPER_SRC = BASE / "mzreflect_wrapper.cpp"
BRIDGE_NAME = "libmzreflect_bridge.dylib"
COMPARE_CSV = OUTPUT_DIR / "ps96_slab_pair_line_reflection_compare.csv"
DEFAULT_FIXED_SCAN_CSV = BASE / "data" / "ps96_slab_pair_line_theta_0.02_1.0_log50.csv"


@dataclass(frozen=True)
class ReflectionConfig(ScanConfig):
    reflector_abundance: float = 1.0
    reflector_iron_abundance: float = 1.0
    hemisphere_mu_order: int = 12
    compare_output_csv: pathlib.Path = COMPARE_CSV


@dataclass(frozen=True)
class IonizedReflectionConfig:
    disk_temperature_k: float
    ionization_parameter: float
    reflector_abundance: float = 1.0
    reflector_iron_abundance: float = 1.0
    hemisphere_mu_order: int = 12


def _candidate_heasoft_roots() -> list[pathlib.Path]:
    candidates: list[pathlib.Path] = []
    env_headas = os.environ.get("HEADAS")
    if env_headas:
        candidates.append(pathlib.Path(env_headas))

    env_conda = os.environ.get("CONDA_PREFIX")
    if env_conda:
        candidates.append(pathlib.Path(env_conda) / "heasoft")

    candidates.extend(
        [
            pathlib.Path("/Users/epiphyllum/anaconda3/envs/heasoft_full/heasoft"),
            pathlib.Path("/Users/epiphyllum/anaconda3/envs/henv/heasoft"),
        ]
    )

    unique: list[pathlib.Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def resolve_heasoft_root() -> pathlib.Path:
    for root in _candidate_heasoft_roots():
        if (root / "include" / "XSFunctions" / "MZCompRefl.h").exists() and (root / "lib" / "libXSFunctions.dylib").exists():
            return root
    raise RuntimeError("Unable to locate a HEASoft installation with the XSPEC reflect libraries.")


def build_reflect_bridge() -> pathlib.Path:
    BUILD_DIR.mkdir(exist_ok=True)
    heasoft_root = resolve_heasoft_root()
    output = BUILD_DIR / BRIDGE_NAME

    if output.exists() and output.stat().st_mtime >= WRAPPER_SRC.stat().st_mtime:
        linked_libraries = subprocess.run(
            ["otool", "-L", str(output)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        expected_library = str(heasoft_root / "lib" / "libXSFunctions.dylib")
        if expected_library in linked_libraries:
            return output

    conda_lib = heasoft_root.parent / "lib"
    cmd = [
        "clang++",
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-dynamiclib",
        "-I",
        str(heasoft_root / "include"),
        "-I",
        str(heasoft_root / "include" / "XSFunctions"),
        str(WRAPPER_SRC),
        "-L",
        str(heasoft_root / "lib"),
        "-L",
        str(conda_lib),
        "-lXSFunctions",
        "-lXSUtil",
        "-lXSModel",
        "-Wl,-rpath," + str(heasoft_root / "lib"),
        "-Wl,-rpath," + str(conda_lib),
        "-o",
        str(output),
    ]
    subprocess.run(cmd, check=True, cwd=BUILD_DIR)
    return output


class NeutralReflectionKernel:
    def __init__(self, config: ReflectionConfig):
        self.config = config
        os.environ.setdefault("HEADAS", str(resolve_heasoft_root() / "bin"))
        self._bridge = ctypes.CDLL(str(build_reflect_bridge()))
        self._bridge.mz_reflect_spectrum.argtypes = [
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ]
        self._bridge.mz_reflect_spectrum.restype = ctypes.c_int

        nodes, weights = np.polynomial.legendre.leggauss(config.hemisphere_mu_order)
        self.mu_nodes = np.ascontiguousarray(0.5 * (nodes + 1.0), dtype=np.float64)
        self.mu_weights = np.ascontiguousarray(0.5 * weights, dtype=np.float64)

    def reflection_only(self, x_grid: np.ndarray, incident_spectrum: np.ndarray, mu_obs: float) -> np.ndarray:
        x = np.ascontiguousarray(np.asarray(x_grid, dtype=np.float64))
        spinc = np.ascontiguousarray(np.asarray(incident_spectrum, dtype=np.float64))
        spref = np.zeros_like(spinc)
        status = self._bridge.mz_reflect_spectrum(
            int(x.size),
            x,
            spinc,
            float(mu_obs),
            float(self.config.reflector_abundance),
            float(self.config.reflector_iron_abundance),
            float(x[-1]),
            spref,
        )
        if status != 0:
            raise RuntimeError(f"HEASoft reflect bridge failed with status={status}.")
        return spref

    def hemisphere_response(
        self,
        x_grid: np.ndarray,
        x_weights: np.ndarray,
        incident_spectrum: np.ndarray,
        observer_mu: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        observer_spectrum = self.reflection_only(x_grid, incident_spectrum, observer_mu)
        observer_flux = float(np.sum((x_grid * x_weights) * observer_spectrum))

        hemisphere_spectrum = np.zeros_like(observer_spectrum)
        for mu_obs, mu_weight in zip(self.mu_nodes, self.mu_weights):
            hemisphere_spectrum += mu_weight * self.reflection_only(x_grid, incident_spectrum, float(mu_obs))

        hemisphere_flux = float(np.sum((x_grid * x_weights) * hemisphere_spectrum))
        return observer_spectrum, hemisphere_spectrum, observer_flux, hemisphere_flux


class IonizedReflectionKernel(NeutralReflectionKernel):
    def __init__(self, config: IonizedReflectionConfig):
        super().__init__(config)
        self.config = config
        self._bridge.mz_ireflect_spectrum.argtypes = [
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ]
        self._bridge.mz_ireflect_spectrum.restype = ctypes.c_int

    def reflection_only(self, x_grid: np.ndarray, incident_spectrum: np.ndarray, mu_obs: float) -> np.ndarray:
        x = np.ascontiguousarray(np.asarray(x_grid, dtype=np.float64))
        spinc = np.ascontiguousarray(np.asarray(incident_spectrum, dtype=np.float64))
        spref = np.zeros_like(spinc)
        status = self._bridge.mz_ireflect_spectrum(
            int(x.size),
            x,
            spinc,
            float(mu_obs),
            float(self.config.reflector_abundance),
            float(self.config.reflector_iron_abundance),
            float(self.config.disk_temperature_k),
            float(self.config.ionization_parameter),
            float(x[-1]),
            spref,
        )
        if status != 0:
            raise RuntimeError(f"HEASoft ireflect bridge failed with status={status}.")
        return spref


class ReflectionCoupledSlabSolver(ComppsSlabSolver):
    def __init__(self, config: ReflectionConfig):
        super().__init__(config)
        self.config = config
        self.reflector = NeutralReflectionKernel(config)
        self._reflection_cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray, float, float, float]] = {}

    def reflection_state(self, theta: float, tau_t: float) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        cache_key = self._round_key(theta, tau_t)
        if cache_key in self._reflection_cache:
            return self._reflection_cache[cache_key]

        state = self.run_state(theta, tau_t)
        observer_spectrum, hemisphere_spectrum, observer_flux, hemisphere_flux = self.reflector.hemisphere_response(
            state.x_grid,
            state.x_weights,
            state.comp_down_spectrum_model,
            self.config.observer_mu,
        )
        effective_albedo = hemisphere_flux / max(state.comp_down_flux_model, 1.0e-30)
        effective_albedo = min(max(effective_albedo, 0.0), 0.999999)

        result = (observer_spectrum, hemisphere_spectrum, observer_flux, hemisphere_flux, effective_albedo)
        self._reflection_cache[cache_key] = result
        return result

    def energy_terms(self, state: RadiativeState, effective_albedo: float) -> tuple[float, float, float]:
        lc_over_ldiss = 1.0 / max(1.0 - state.p_sc * state.eta, 1.0e-12)
        ls_over_ldiss = (1.0 - effective_albedo) * state.eta * lc_over_ldiss
        amplification_required = 1.0 / max((1.0 - effective_albedo) * state.eta, 1.0e-12)
        return lc_over_ldiss, ls_over_ldiss, amplification_required

    def energy_residual(self, theta: float, tau_t: float) -> float:
        state = self.run_state(theta, tau_t)
        _, _, _, _, effective_albedo = self.reflection_state(theta, tau_t)
        _, _, amplification_required = self.energy_terms(state, effective_albedo)
        return math.log(state.amplification_model / amplification_required)

    def pair_production_rate_per_ldiss2(self, state: RadiativeState, ls_over_ldiss: float) -> float:
        flux_scale = ls_over_ldiss * MEC2_ERG * CLIGHT / (SIGMA_T * SLAB_HEIGHT_CM * state.seed_flux_model)
        field_physical = state.internal_field_model * flux_scale
        kernel = self._ensure_kernel(state)
        return kernel.pair_production_rate(field_physical, state.tau_grid)

    def solve_point(self, theta: float, guess_tau: float | None) -> dict[str, float | str]:
        tau_t, root_method = self.find_tau_root(theta, guess_tau)
        state = self.run_state(theta, tau_t)
        observer_spectrum, _, observer_flux, hemisphere_flux, effective_albedo = self.reflection_state(theta, tau_t)
        lc_over_ldiss, ls_over_ldiss, amplification_required = self.energy_terms(state, effective_albedo)
        prod_coeff = self.pair_production_rate_per_ldiss2(state, ls_over_ldiss)
        ann_rate = self.pair_annihilation_rate(theta, tau_t)
        ldiss = math.sqrt(ann_rate / prod_coeff)

        return {
            "theta": theta,
            "kTe_keV": theta / KTE_TO_THETA,
            "tau_T": tau_t,
            "l_diss_local": ldiss,
            "eta": state.eta,
            "p_sc": state.p_sc,
            "A_model": state.amplification_model,
            "A_required": amplification_required,
            "effective_albedo": effective_albedo,
            "reflected_flux_model_observer": observer_flux,
            "reflected_flux_model_hemisphere": hemisphere_flux,
            "downward_flux_model": state.comp_down_flux_model,
            "l_s_over_l_diss": ls_over_ldiss,
            "l_c_over_l_diss": lc_over_ldiss,
            "pair_production_rate_unit_ldiss2": prod_coeff,
            "pair_annihilation_rate": ann_rate,
            "energy_log_residual": math.log(state.amplification_model / amplification_required),
            "root_method": root_method,
            "observed_total_up_flux_model": state.comp_up_flux_model + observer_flux,
            "reflection_to_corona_ratio": hemisphere_flux / max(state.comp_down_flux_model, 1.0e-30),
            "reflection_to_observer_ratio": observer_flux / max(state.comp_up_flux_model, 1.0e-30),
            "observer_reflection_peak_x": float(state.x_grid[int(np.argmax(observer_spectrum))]),
        }


def scan_reflection_theta_grid(config: ReflectionConfig) -> list[dict[str, float | str]]:
    solver = ReflectionCoupledSlabSolver(config)
    rows: list[dict[str, float | str]] = []
    previous_tau: float | None = None

    for theta in logspace(config.theta_min, config.theta_max, config.n_samples):
        row = solver.solve_point(theta, previous_tau)
        rows.append(row)
        previous_tau = float(row["tau_T"])

    return rows


def compare_fixed_and_reflection(config: ReflectionConfig) -> list[dict[str, float | str]]:
    fixed_solver = ComppsSlabSolver(config)
    reflect_solver = ReflectionCoupledSlabSolver(config)
    rows: list[dict[str, float | str]] = []
    previous_fixed_tau: float | None = None
    previous_reflect_tau: float | None = None

    for theta in logspace(config.theta_min, config.theta_max, config.n_samples):
        fixed_row = fixed_solver.solve_point(theta, previous_fixed_tau)
        reflect_row = reflect_solver.solve_point(theta, previous_reflect_tau)
        previous_fixed_tau = float(fixed_row["tau_T"])
        previous_reflect_tau = float(reflect_row["tau_T"])

        rows.append(
            {
                "theta": theta,
                "kTe_keV": reflect_row["kTe_keV"],
                "eta_fixed": fixed_row["eta"],
                "eta_reflect": reflect_row["eta"],
                "tau_T_fixed": fixed_row["tau_T"],
                "tau_T_reflect": reflect_row["tau_T"],
                "tau_ratio_reflect_to_fixed": float(reflect_row["tau_T"]) / max(float(fixed_row["tau_T"]), 1.0e-30),
                "l_diss_fixed": fixed_row["l_diss_local"],
                "l_diss_reflect": reflect_row["l_diss_local"],
                "ldiss_ratio_reflect_to_fixed": float(reflect_row["l_diss_local"]) / max(float(fixed_row["l_diss_local"]), 1.0e-30),
                "effective_albedo_reflect": reflect_row["effective_albedo"],
                "albedo_fixed": config.albedo,
                "A_required_fixed": fixed_row["A_required"],
                "A_required_reflect": reflect_row["A_required"],
                "reflected_flux_model_observer": reflect_row["reflected_flux_model_observer"],
                "reflected_flux_model_hemisphere": reflect_row["reflected_flux_model_hemisphere"],
                "downward_flux_model": reflect_row["downward_flux_model"],
                "observed_total_up_flux_reflect_model": reflect_row["observed_total_up_flux_model"],
                "energy_log_residual_fixed": fixed_row["energy_log_residual"],
                "energy_log_residual_reflect": reflect_row["energy_log_residual"],
                "root_method_fixed": fixed_row["root_method"],
                "root_method_reflect": reflect_row["root_method"],
            }
        )

    return rows


def load_fixed_scan_rows(csv_path: pathlib.Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            parsed: dict[str, float | str] = {}
            for key, value in raw.items():
                if value is None:
                    parsed[key] = ""
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def compare_existing_fixed_and_reflection(
    config: ReflectionConfig,
    fixed_csv: pathlib.Path = DEFAULT_FIXED_SCAN_CSV,
) -> list[dict[str, float | str]]:
    fixed_rows = load_fixed_scan_rows(fixed_csv)
    reflect_solver = ReflectionCoupledSlabSolver(config)
    rows: list[dict[str, float | str]] = []
    previous_reflect_tau: float | None = None

    for fixed_row in fixed_rows:
        theta = float(fixed_row["theta"])
        reflect_row = reflect_solver.solve_point(theta, previous_reflect_tau)
        previous_reflect_tau = float(reflect_row["tau_T"])

        rows.append(
            {
                "theta": theta,
                "kTe_keV": reflect_row["kTe_keV"],
                "eta_fixed": fixed_row["eta"],
                "eta_reflect": reflect_row["eta"],
                "tau_T_fixed": fixed_row["tau_T"],
                "tau_T_reflect": reflect_row["tau_T"],
                "tau_ratio_reflect_to_fixed": float(reflect_row["tau_T"]) / max(float(fixed_row["tau_T"]), 1.0e-30),
                "l_diss_fixed": fixed_row["l_diss_local"],
                "l_diss_reflect": reflect_row["l_diss_local"],
                "ldiss_ratio_reflect_to_fixed": float(reflect_row["l_diss_local"]) / max(float(fixed_row["l_diss_local"]), 1.0e-30),
                "effective_albedo_reflect": reflect_row["effective_albedo"],
                "albedo_fixed": config.albedo,
                "A_required_fixed": fixed_row["A_required"],
                "A_required_reflect": reflect_row["A_required"],
                "reflected_flux_model_observer": reflect_row["reflected_flux_model_observer"],
                "reflected_flux_model_hemisphere": reflect_row["reflected_flux_model_hemisphere"],
                "downward_flux_model": reflect_row["downward_flux_model"],
                "observed_total_up_flux_reflect_model": reflect_row["observed_total_up_flux_model"],
                "energy_log_residual_fixed": fixed_row["energy_log_residual"],
                "energy_log_residual_reflect": reflect_row["energy_log_residual"],
                "root_method_fixed": fixed_row["root_method"],
                "root_method_reflect": reflect_row["root_method"],
            }
        )

    return rows


def write_comparison_rows(rows: list[dict[str, float | str]], output_csv: pathlib.Path) -> None:
    output_csv.parent.mkdir(exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    config = ReflectionConfig()
    if DEFAULT_FIXED_SCAN_CSV.exists():
        rows = compare_existing_fixed_and_reflection(config, DEFAULT_FIXED_SCAN_CSV)
    else:
        rows = compare_fixed_and_reflection(config)
    write_comparison_rows(rows, config.compare_output_csv)
    print(f"Wrote {len(rows)} rows to {config.compare_output_csv}")


if __name__ == "__main__":
    main()
