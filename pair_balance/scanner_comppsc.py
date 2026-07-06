#!/usr/bin/env python3
"""Slab pair-line solver using the local compPSc transfer implementation."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

ROOT_FOR_IMPORT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from pair_balance.scanner import (
    CLIGHT,
    KTE_TO_THETA,
    MEC2_ERG,
    SIGMA_T,
    SLAB_HEIGHT_CM,
    PairKernel,
    RadiativeState,
    logspace,
)


BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
COMPPSC_DIR = ROOT / "compps_conv"
COMPPSSRC = COMPPSC_DIR / "compps_conv.f"
XWRITE_STUB = COMPPSC_DIR / "xwrite_stub.f"
COMPPSSQ = COMPPSC_DIR / "mscomppsq.inc"
COMPPSSQ_FREE = COMPPSC_DIR / "mscomppsq_free.inc"
BUILD_DIR = BASE / "_build_comppsc"
MODULE_NAME = "_comppsc_pair_balance"
DATA_DIR = BASE / "data"


@dataclass(frozen=True)
class ComppscScanConfig:
    theta_min: float = 0.02
    theta_max: float = 1.0
    n_samples: int = 40
    tau_min: float = 0.005
    tau_max: float = 5.0
    tbb_kev: float = 0.005
    seed_bins: int = 1200
    albedo: float = 0.2
    d_ratio: float = 0.0
    max_scatter: int = 2000
    exact_angles: bool = True
    energy_tolerance: float = 5.0e-4
    tau_bisect_iterations: int = 36
    continuation_expand_factor: float = 1.45
    continuation_expand_steps: int = 10
    global_tau_samples: int = 22
    phi_quadrature_order: int = 48
    observer_mu: float = 0.5


def build_comppsc_extension() -> pathlib.Path:
    BUILD_DIR.mkdir(exist_ok=True)
    sources = [COMPPSSRC, XWRITE_STUB, COMPPSSQ, COMPPSSQ_FREE]
    existing = sorted(BUILD_DIR.glob(f"{MODULE_NAME}*.so"))
    latest_source_mtime = max(path.stat().st_mtime for path in sources)

    if existing and max(path.stat().st_mtime for path in existing) >= latest_source_mtime:
        return max(existing, key=lambda path: path.stat().st_mtime)

    for path in existing:
        path.unlink()

    cmd = [
        sys.executable,
        "-m",
        "numpy.f2py",
        "-c",
        "-m",
        MODULE_NAME,
        f"-I{COMPPSC_DIR}",
        str(COMPPSSRC),
        str(XWRITE_STUB),
    ]
    env = os.environ.copy()
    conda_root = pathlib.Path(sys.executable).resolve().parents[1]
    heasoft_bin = conda_root / "envs" / "heasoft_full" / "bin"
    path_parts = [str(heasoft_bin), str(conda_root / "bin"), env.get("PATH", "")]
    env["PATH"] = os.pathsep.join(part for part in path_parts if part)
    sdk_path = subprocess.run(
        ["xcrun", "--show-sdk-path"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if sdk_path:
        env["SDKROOT"] = sdk_path
        env["CONDA_BUILD_SYSROOT"] = sdk_path
    subprocess.run(cmd, cwd=BUILD_DIR, check=True, env=env)

    built = sorted(BUILD_DIR.glob(f"{MODULE_NAME}*.so"))
    if not built:
        raise RuntimeError("f2py build completed without producing a compPSc extension.")
    return max(built, key=lambda path: path.stat().st_mtime)


def load_comppsc_module():
    module_path = build_comppsc_extension()
    spec = importlib.util.spec_from_file_location(MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load extension module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_blackbody_seed_grid(tbb_kev: float, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.logspace(-8.0, 2.0, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = np.diff(edges)
    x = centers / tbb_kev
    spec = np.zeros(n_bins, dtype=np.float64)
    low = x < 1.0e-8
    normal = (x >= 1.0e-8) & (x < 700.0)
    spec[low] = centers[low] * tbb_kev * widths[low]
    spec[normal] = centers[normal] * centers[normal] / np.expm1(x[normal]) * widths[normal]
    return edges.astype(np.float64), spec.astype(np.float64)


class ComppscSlabSolver:
    def __init__(self, config: ComppscScanConfig):
        self.config = config
        self.module = load_comppsc_module()
        self.obj_grid = np.logspace(-8.0, 1.0, 4096)
        self.seed_edges, self.seed_spec = build_blackbody_seed_grid(config.tbb_kev, config.seed_bins)
        self._state_cache: dict[tuple[float, float], RadiativeState] = {}
        self._kernel: PairKernel | None = None

    @staticmethod
    def _round_key(theta: float, tau_t: float) -> tuple[float, float]:
        return (round(theta, 12), round(tau_t, 12))

    @staticmethod
    def _safe_positive(value: float, floor: float = 1.0e-12) -> float:
        return max(value, floor)

    @staticmethod
    def _angular_flux(flat_field: np.ndarray, mu: np.ndarray, w: np.ndarray) -> np.ndarray:
        n_energy = int(flat_field.size // mu.size)
        reshaped = np.array(flat_field, dtype=float).reshape((n_energy, mu.size), order="C")
        return np.sum(reshaped * (w[None, :] * mu[None, :]), axis=1)

    def _ensure_kernel(self, state: RadiativeState) -> PairKernel:
        if self._kernel is None:
            self._kernel = PairKernel(
                x_grid=state.x_grid,
                x_weights=state.x_weights,
                mu_grid_full=state.mu_grid_full,
                mu_weights_full=state.mu_weights_full,
                phi_order=self.config.phi_quadrature_order,
            )
        return self._kernel

    def _configure_transfer(self) -> None:
        if hasattr(self.module, "msmaxscovr"):
            self.module.msmaxscovr.maxsc_override = int(self.config.max_scatter)
        if hasattr(self.module, "msqxact"):
            self.module.msqxact.iexactang = 1 if self.config.exact_angles else 0

    def _run_msismco(self, parm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phcon = np.zeros_like(self.obj_grid)
        phblb = np.zeros_like(self.obj_grid)
        phref = np.zeros_like(self.obj_grid)
        phnorm = np.array(1.0, dtype=np.float64)
        self.module.msismco(
            parm,
            self.obj_grid,
            phcon,
            phblb,
            phref,
            phnorm,
            True,
            True,
            self.seed_spec,
            self.seed_edges,
        )
        return phcon, phblb, phref

    def run_state(self, theta: float, tau_t: float) -> RadiativeState:
        cache_key = self._round_key(theta, tau_t)
        if cache_key in self._state_cache:
            return self._state_cache[cache_key]

        self._configure_transfer()
        parm = np.array(
            [
                theta / KTE_TO_THETA,
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
        self._run_msismco(parm)

        x_grid = np.array(self.module.msqqwfre.xen, dtype=float)
        x_weights = np.array(self.module.msqqcm2.a, dtype=float)
        mu = np.array(self.module.msqqcm2.uang, dtype=float)
        mu_w = np.array(self.module.msqqcm2.aang, dtype=float)
        mu_grid_full = np.concatenate((-mu[::-1], mu))
        mu_weights_full = np.concatenate((mu_w[::-1], mu_w))

        seed_flux = float(np.sum((x_grid * x_weights) * np.array(self.module.msqbb.dinten, dtype=float)))
        unscattered_spectrum = self._angular_flux(self.module.msqqres.dintpl, mu, mu_w)
        comp_up_spectrum = self._angular_flux(self.module.msqqres.suipl, mu, mu_w)
        comp_down_spectrum = self._angular_flux(self.module.msqqres.suimi, mu, mu_w)
        unscattered_flux = float(np.sum((x_grid * x_weights) * unscattered_spectrum))
        comp_up_flux = float(np.sum((x_grid * x_weights) * comp_up_spectrum))
        comp_down_flux = float(np.sum((x_grid * x_weights) * comp_down_spectrum))

        comp_total = comp_up_flux + comp_down_flux
        eta = comp_down_flux / self._safe_positive(comp_total)
        p_sc = 1.0 - unscattered_flux / self._safe_positive(seed_flux)
        amplification_model = comp_total / self._safe_positive(seed_flux)
        amplification_required = self.amplification_required_from_terms(eta, p_sc)
        ls_over_ldiss, lc_over_ldiss, _, _ = self.compactness_terms_from_terms(eta, p_sc)

        n_tau = np.array(self.module.msqqdep.taugrid, dtype=float).size
        n_energy = x_grid.size
        n_mu = mu.size
        tintpl = np.array(self.module.msqqint.tintpl, dtype=float).T.reshape((n_tau, n_energy, n_mu), order="C")
        tintmi = np.array(self.module.msqqint.tintmi, dtype=float).T.reshape((n_tau, n_energy, n_mu), order="C")
        internal_field = np.concatenate((tintmi[:, :, ::-1], tintpl), axis=2)
        tau_grid = np.array(self.module.msqqdep.taugrid, dtype=float)

        state = RadiativeState(
            theta=theta,
            tau_t=tau_t,
            seed_flux_model=seed_flux,
            unscattered_flux_model=unscattered_flux,
            comp_up_flux_model=comp_up_flux,
            comp_down_flux_model=comp_down_flux,
            unscattered_spectrum_model=unscattered_spectrum,
            comp_up_spectrum_model=comp_up_spectrum,
            comp_down_spectrum_model=comp_down_spectrum,
            eta=eta,
            p_sc=p_sc,
            amplification_model=amplification_model,
            amplification_required=amplification_required,
            ls_over_ldiss=ls_over_ldiss,
            lc_over_ldiss=lc_over_ldiss,
            tau_grid=tau_grid,
            internal_field_model=internal_field,
            x_grid=x_grid,
            x_weights=x_weights,
            mu_grid_full=mu_grid_full,
            mu_weights_full=mu_weights_full,
        )
        self._state_cache[cache_key] = state
        return state

    def compactness_terms_from_terms(self, eta: float, p_sc: float) -> tuple[float, float, float, float]:
        transport_denom = self._safe_positive(1.0 - p_sc * eta)
        lc_over_ldiss = (1.0 + self.config.d_ratio * p_sc) / transport_denom
        reprocessed_seed = (1.0 - self.config.albedo) * eta * lc_over_ldiss
        intrinsic_seed = self.config.d_ratio
        ls_over_ldiss = intrinsic_seed + reprocessed_seed
        return ls_over_ldiss, lc_over_ldiss, intrinsic_seed, reprocessed_seed

    def amplification_required_from_terms(self, eta: float, p_sc: float) -> float:
        feedback = (1.0 - self.config.albedo) * eta
        disk_term = self.config.d_ratio * (1.0 - self.config.albedo * eta * p_sc)
        return (1.0 + self.config.d_ratio * p_sc) / self._safe_positive(feedback + disk_term)

    def energy_residual(self, theta: float, tau_t: float) -> float:
        state = self.run_state(theta, tau_t)
        return math.log(state.amplification_model / self._safe_positive(state.amplification_required))

    def _bisection(self, theta: float, lo: float, hi: float) -> float:
        f_lo = self.energy_residual(theta, lo)
        f_hi = self.energy_residual(theta, hi)
        if f_lo * f_hi > 0.0:
            raise RuntimeError("Bisection requested without a sign change.")

        for _ in range(self.config.tau_bisect_iterations):
            mid = math.sqrt(lo * hi)
            f_mid = self.energy_residual(theta, mid)
            if abs(f_mid) < self.config.energy_tolerance:
                return mid
            if f_lo * f_mid <= 0.0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid
        return math.sqrt(lo * hi)

    def _global_roots(self, theta: float) -> list[float]:
        tau_grid = np.geomspace(self.config.tau_min, self.config.tau_max, self.config.global_tau_samples)
        residuals = [self.energy_residual(theta, float(tau_t)) for tau_t in tau_grid]
        roots: list[float] = []

        for idx in range(len(tau_grid) - 1):
            f_lo = residuals[idx]
            f_hi = residuals[idx + 1]
            if abs(f_lo) < self.config.energy_tolerance:
                roots.append(float(tau_grid[idx]))
                continue
            if f_lo * f_hi > 0.0:
                continue
            roots.append(self._bisection(theta, float(tau_grid[idx]), float(tau_grid[idx + 1])))

        deduped: list[float] = []
        for root in roots:
            if not deduped or abs(math.log(root / deduped[-1])) > 1.0e-6:
                deduped.append(root)
        return deduped

    def find_tau_root(self, theta: float, guess_tau: float | None) -> tuple[float, str]:
        if guess_tau is not None:
            guess_tau = min(max(guess_tau, self.config.tau_min), self.config.tau_max)
            guess_residual = self.energy_residual(theta, guess_tau)
            if abs(guess_residual) < self.config.energy_tolerance:
                return guess_tau, "continuation"

            lower = guess_tau
            upper = guess_tau
            lower_residual = guess_residual
            upper_residual = guess_residual
            factor = self.config.continuation_expand_factor
            for _ in range(self.config.continuation_expand_steps):
                if lower > self.config.tau_min:
                    lower = max(lower / factor, self.config.tau_min)
                    lower_residual = self.energy_residual(theta, lower)
                if lower_residual * guess_residual <= 0.0:
                    return self._bisection(theta, lower, guess_tau), "continuation"

                if upper < self.config.tau_max:
                    upper = min(upper * factor, self.config.tau_max)
                    upper_residual = self.energy_residual(theta, upper)
                if upper_residual * guess_residual <= 0.0:
                    return self._bisection(theta, guess_tau, upper), "continuation"

        roots = self._global_roots(theta)
        if not roots:
            raise RuntimeError(f"No compPSc slab energy-balance root found for theta={theta:.6g}.")
        if guess_tau is None:
            return max(roots), "global"
        return min(roots, key=lambda root: abs(math.log(root / guess_tau))), "global"

    def pair_production_rate_per_ldiss2(self, state: RadiativeState, ls_over_ldiss: float) -> float:
        flux_scale = ls_over_ldiss * MEC2_ERG * CLIGHT / (
            SIGMA_T * SLAB_HEIGHT_CM * self._safe_positive(state.seed_flux_model)
        )
        field_physical = state.internal_field_model * flux_scale
        kernel = self._ensure_kernel(state)
        return kernel.pair_production_rate(field_physical, state.tau_grid)

    @staticmethod
    def pair_annihilation_rate(theta: float, tau_t: float) -> float:
        from pair_balance.scanner import R_E, EULER_ETA

        log_term = math.log(1.3 + 2.0 * EULER_ETA * theta)
        ann_shape = math.pi / (1.0 + 2.0 * theta * theta / max(log_term, 1.0e-12))
        n_species = tau_t / (2.0 * SIGMA_T * SLAB_HEIGHT_CM)
        return n_species * n_species * CLIGHT * R_E * R_E * ann_shape

    def solve_point(self, theta: float, guess_tau: float | None) -> dict[str, float | str | int | bool]:
        tau_t, root_method = self.find_tau_root(theta, guess_tau)
        state = self.run_state(theta, tau_t)
        ls_over_ldiss, lc_over_ldiss, intrinsic_seed, reprocessed_seed = self.compactness_terms_from_terms(
            state.eta, state.p_sc
        )
        prod_coeff = self.pair_production_rate_per_ldiss2(state, ls_over_ldiss)
        ann_rate = self.pair_annihilation_rate(theta, tau_t)
        ldiss = math.sqrt(ann_rate / self._safe_positive(prod_coeff))
        last_scatter = int(getattr(self.module.msiterstat, "lastisc", -1)) if hasattr(self.module, "msiterstat") else -1
        last_difmax = (
            float(getattr(self.module.msiterstat, "lastdifmax", math.nan)) if hasattr(self.module, "msiterstat") else math.nan
        )
        return {
            "model": "compPSc",
            "d_ratio": self.config.d_ratio,
            "theta": theta,
            "kTe_keV": theta / KTE_TO_THETA,
            "tau_T": tau_t,
            "l_diss_local": ldiss,
            "eta": state.eta,
            "p_sc": state.p_sc,
            "A_model": state.amplification_model,
            "A_required": state.amplification_required,
            "l_s_over_l_diss": ls_over_ldiss,
            "l_c_over_l_diss": lc_over_ldiss,
            "intrinsic_seed_over_l_diss": intrinsic_seed,
            "reprocessed_seed_over_l_diss": reprocessed_seed,
            "pair_production_rate_unit_ldiss2": prod_coeff,
            "pair_annihilation_rate": ann_rate,
            "energy_log_residual": self.energy_residual(theta, tau_t),
            "root_method": root_method,
            "max_scatter": self.config.max_scatter,
            "last_scatter_order": last_scatter,
            "last_difmax": last_difmax,
            "exact_angles": self.config.exact_angles,
            "max_tau_grid": int(state.tau_grid.size),
            "max_frequency_grid": int(state.x_grid.size),
            "seed_bins": self.config.seed_bins,
        }


def scan_single(config: ComppscScanConfig) -> list[dict[str, float | str | int | bool]]:
    solver = ComppscSlabSolver(config)
    rows: list[dict[str, float | str | int | bool]] = []
    previous_tau: float | None = None
    theta_values = [config.theta_min] if config.n_samples == 1 else logspace(config.theta_min, config.theta_max, config.n_samples)
    for theta in theta_values:
        row = solver.solve_point(theta, previous_tau)
        rows.append(row)
        previous_tau = float(row["tau_T"])
        print(
            "d={d_ratio:g} theta={theta:.5g} tau={tau_T:.5g} "
            "ldiss={l_diss_local:.5g} sc={last_scatter_order} dif={last_difmax:.3g}".format(**row),
            flush=True,
        )
    return rows


def write_rows(rows: list[dict[str, float | str | int | bool]], output_csv: pathlib.Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty scan")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta-min", type=float, default=0.02)
    parser.add_argument("--theta-max", type=float, default=1.0)
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--max-scatter", type=int, default=2000)
    parser.add_argument("--seed-bins", type=int, default=1200)
    parser.add_argument("--d-ratios", type=float, nargs="+", default=[0.0, 0.25, 1.0, 3.0])
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--no-exact-angles", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        suffix = f"theta_{args.theta_min:g}_{args.theta_max:g}_log{args.n_samples}_maxsc{args.max_scatter}"
        output = DATA_DIR / f"comppsc_slab_dratio_tau_kTe_{suffix}.csv"

    rows: list[dict[str, float | str | int | bool]] = []
    for d_ratio in args.d_ratios:
        config = ComppscScanConfig(
            theta_min=args.theta_min,
            theta_max=args.theta_max,
            n_samples=args.n_samples,
            d_ratio=float(d_ratio),
            max_scatter=args.max_scatter,
            seed_bins=args.seed_bins,
            exact_angles=not args.no_exact_angles,
        )
        rows.extend(scan_single(config))
        write_rows(rows, output)
        print(f"Checkpoint: wrote {len(rows)} rows to {output}", flush=True)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    main()
