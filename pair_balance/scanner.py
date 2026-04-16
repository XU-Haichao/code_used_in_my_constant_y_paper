#!/usr/bin/env python3
"""Direct slab pair-line solver based on COMPPS radiative transfer.

This scanner follows the three-step logic described for slabs in Stern et al.
(1995) and Poutanen & Svensson (1996):

1. For fixed (theta, tau_T), run the exact iterative-scattering transfer solver
   in slab geometry and recover the emergent hard flux, the downward feedback,
   the unscattered blackbody flux, and the full depth-angle-energy radiation
   field inside the slab.
2. Solve the slab energy balance for tau_T at fixed theta using the PS96
   equations for d=0, g=1 and fixed cold-disk albedo.
3. At the energy-balance root, compute the depth-averaged gamma-gamma pair
   production rate from the exported internal field and close ldiss from pair
   balance against thermal pair annihilation.

The local COMPPS patch used here does two things needed by the solver:

- expose the exact internal intensity field through the QQINT/QQDEP common
  blocks; and
- allow forcing exact anisotropy for all scattering orders through QXACT.

The current closure is for the pure-pair slab discussed in Stern et al. (1995):
Tbb = 5 eV, no intrinsic cold-disk dissipation, and bottom injection of the
reprocessed soft photons.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import pathlib
import subprocess
import sys
from dataclasses import dataclass

import numpy as np


BASE = pathlib.Path(__file__).resolve().parent
COMPPSSRC = BASE / "compps.f"
COMPPSSQ = BASE / "comppsq.inc"
COMPPSJ = BASE / "comppsj.inc"
XWRITE_STUB = BASE / "xwrite_stub.f"
BUILD_DIR = BASE / "_build"
MODULE_NAME = "_compps_pair_balance"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_CSV = OUTPUT_DIR / "ps96_slab_pair_line_scan.csv"
OUTPUT_NOTES = OUTPUT_DIR / "ps96_slab_pair_line_scan_notes.txt"

SIGMA_T = 6.6524587321e-25
MEC2_ERG = 8.1871057769e-7
CLIGHT = 2.99792458e10
R_E = 2.8179403262e-13
EULER_ETA = 0.5615
KTE_TO_THETA = 1.0 / 511.0
DEFAULT_TBB_KEV = 0.005
SLAB_HEIGHT_CM = 1.0


@dataclass(frozen=True)
class ScanConfig:
    theta_min: float = 0.0028016488
    theta_max: float = 1.80
    n_samples: int = 40
    tbb_kev: float = DEFAULT_TBB_KEV
    albedo: float = 0.2
    tau_min: float = 0.01
    tau_max: float = 3.0
    energy_tolerance: float = 5.0e-4
    tau_bisect_iterations: int = 40
    continuation_expand_factor: float = 1.4
    continuation_expand_steps: int = 10
    global_tau_samples: int = 24
    phi_quadrature_order: int = 48
    observer_mu: float = 0.5


@dataclass
class RadiativeState:
    theta: float
    tau_t: float
    seed_flux_model: float
    unscattered_flux_model: float
    comp_up_flux_model: float
    comp_down_flux_model: float
    unscattered_spectrum_model: np.ndarray
    comp_up_spectrum_model: np.ndarray
    comp_down_spectrum_model: np.ndarray
    eta: float
    p_sc: float
    amplification_model: float
    amplification_required: float
    ls_over_ldiss: float
    lc_over_ldiss: float
    tau_grid: np.ndarray
    internal_field_model: np.ndarray
    x_grid: np.ndarray
    x_weights: np.ndarray
    mu_grid_full: np.ndarray
    mu_weights_full: np.ndarray


def logspace(start: float, stop: float, count: int) -> list[float]:
    log_start = math.log10(start)
    log_stop = math.log10(stop)
    return [10 ** (log_start + i * (log_stop - log_start) / (count - 1)) for i in range(count)]


def build_compps_extension() -> pathlib.Path:
    BUILD_DIR.mkdir(exist_ok=True)
    source_paths = [COMPPSSRC, COMPPSSQ, COMPPSJ, XWRITE_STUB]
    existing = sorted(BUILD_DIR.glob(f"{MODULE_NAME}*.so"))
    latest_source_mtime = max(path.stat().st_mtime for path in source_paths)

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
        f"-I{BASE}",
        str(COMPPSSRC),
        str(XWRITE_STUB),
    ]
    subprocess.run(cmd, cwd=BUILD_DIR, check=True)

    built = sorted(BUILD_DIR.glob(f"{MODULE_NAME}*.so"))
    if not built:
        raise RuntimeError("f2py build completed without producing a COMPPS extension.")
    return max(built, key=lambda path: path.stat().st_mtime)


def load_compps_module():
    module_path = build_compps_extension()
    spec = importlib.util.spec_from_file_location(MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load extension module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PairKernel:
    def __init__(self, x_grid: np.ndarray, x_weights: np.ndarray, mu_grid_full: np.ndarray, mu_weights_full: np.ndarray, phi_order: int):
        phi_nodes, phi_weights = np.polynomial.legendre.leggauss(phi_order)
        phi = math.pi * (phi_nodes + 1.0)
        phi_weights = math.pi * phi_weights

        sin_mu = np.sqrt(np.maximum(0.0, 1.0 - mu_grid_full * mu_grid_full))
        cos_psi = (
            mu_grid_full[:, None, None] * mu_grid_full[None, :, None]
            + sin_mu[:, None, None] * sin_mu[None, :, None] * np.cos(phi)[None, None, :]
        )
        one_minus_cos = 1.0 - cos_psi

        x_product = x_grid[:, None, None, None, None] * x_grid[None, :, None, None, None]
        s_param = 0.5 * x_product * one_minus_cos[None, None, :, :, :]
        valid = s_param > 1.0

        beta = np.zeros_like(s_param)
        beta[valid] = np.sqrt(1.0 - 1.0 / s_param[valid])

        sigma_gg = np.zeros_like(s_param)
        beta_valid = beta[valid]
        sigma_gg[valid] = (3.0 * SIGMA_T / 16.0) * (1.0 - beta_valid * beta_valid) * (
            (3.0 - beta_valid**4) * np.log((1.0 + beta_valid) / (1.0 - beta_valid))
            - 2.0 * beta_valid * (2.0 - beta_valid * beta_valid)
        )

        kernel = np.tensordot(one_minus_cos * sigma_gg, phi_weights, axes=([4], [0]))
        weights = (
            x_weights[:, None, None, None]
            * x_weights[None, :, None, None]
            * mu_weights_full[None, None, :, None]
            * mu_weights_full[None, None, None, :]
        )
        self.weighted_kernel = kernel * weights
        self.prefactor = 1.0 / (4.0 * math.pi * MEC2_ERG * MEC2_ERG * CLIGHT)

    def pair_production_rate(self, field_physical: np.ndarray, tau_grid: np.ndarray) -> float:
        rate_depth = np.array(
            [
                self.prefactor * np.einsum("ia,jb,ijab->", slab_field, slab_field, self.weighted_kernel, optimize=True)
                for slab_field in field_physical
            ],
            dtype=float,
        )
        if field_physical.shape[0] == 1:
            return float(rate_depth[0])
        return float(np.trapezoid(rate_depth, tau_grid))


class ComppsSlabSolver:
    def __init__(self, config: ScanConfig):
        self.config = config
        self.module = load_compps_module()
        self.module.qxact.iexactang = 1
        self.obj_grid = np.logspace(-8.0, 1.0, 4096)
        self._tmp_ph = np.zeros_like(self.obj_grid)
        self._state_cache: dict[tuple[float, float], RadiativeState] = {}
        self._kernel: PairKernel | None = None

    @staticmethod
    def _round_key(theta: float, tau_t: float) -> tuple[float, float]:
        return (round(theta, 12), round(tau_t, 12))

    @staticmethod
    def _angular_flux(flat_field: np.ndarray, mu: np.ndarray, w: np.ndarray) -> np.ndarray:
        reshaped = np.array(flat_field, dtype=float).reshape((88, 5), order="C")
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

    def run_state(self, theta: float, tau_t: float) -> RadiativeState:
        cache_key = self._round_key(theta, tau_t)
        if cache_key in self._state_cache:
            return self._state_cache[cache_key]

        parm = np.array(
            [
                theta / KTE_TO_THETA,
                0.0,
                0.0,
                0.0,
                self.config.tbb_kev,
                tau_t,
                1.0,
                0.0,
                self.config.observer_mu,
                1.0,
                0.0,
            ],
            dtype=float,
        )

        self.module.qxact.iexactang = 1
        self.module.ismco(parm, self.obj_grid, self._tmp_ph, self._tmp_ph, self._tmp_ph, 1.0, 1)

        x_grid = np.array(self.module.qqwfre.xen, dtype=float)
        x_weights = np.array(self.module.qqcm2.a, dtype=float)
        mu = np.array(self.module.qqcm2.uang, dtype=float)
        mu_w = np.array(self.module.qqcm2.aang, dtype=float)
        mu_grid_full = np.concatenate((-mu[::-1], mu))
        mu_weights_full = np.concatenate((mu_w[::-1], mu_w))

        seed_flux = float(np.sum((x_grid * x_weights) * np.array(self.module.qbb.dinten, dtype=float)))
        unscattered_spectrum = self._angular_flux(self.module.qqres.dintpl, mu, mu_w)
        comp_up_spectrum = self._angular_flux(self.module.qqres.suipl, mu, mu_w)
        comp_down_spectrum = self._angular_flux(self.module.qqres.suimi, mu, mu_w)
        unscattered_flux = float(np.sum((x_grid * x_weights) * unscattered_spectrum))
        comp_up_flux = float(np.sum((x_grid * x_weights) * comp_up_spectrum))
        comp_down_flux = float(np.sum((x_grid * x_weights) * comp_down_spectrum))

        comp_total = comp_up_flux + comp_down_flux
        eta = comp_down_flux / comp_total
        p_sc = 1.0 - unscattered_flux / seed_flux
        amplification_model = comp_total / seed_flux
        amplification_required = 1.0 / ((1.0 - self.config.albedo) * eta)

        lc_over_ldiss = 1.0 / max(1.0 - p_sc * eta, 1.0e-12)
        ls_over_ldiss = (1.0 - self.config.albedo) * eta * lc_over_ldiss

        tintpl = np.array(self.module.qqint.tintpl, dtype=float).T.reshape((25, 88, 5), order="C")
        tintmi = np.array(self.module.qqint.tintmi, dtype=float).T.reshape((25, 88, 5), order="C")
        internal_field = np.concatenate((tintmi[:, :, ::-1], tintpl), axis=2)
        tau_grid = np.array(self.module.qqdep.taugrid, dtype=float)

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

    def energy_residual(self, theta: float, tau_t: float) -> float:
        state = self.run_state(theta, tau_t)
        return math.log(state.amplification_model / state.amplification_required)

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
            raise RuntimeError(f"No slab energy-balance root found for theta={theta:.6g}.")

        if guess_tau is None:
            return max(roots), "global"
        return min(roots, key=lambda root: abs(math.log(root / guess_tau))), "global"

    def pair_production_rate_per_ldiss2(self, state: RadiativeState) -> float:
        flux_scale = state.ls_over_ldiss * MEC2_ERG * CLIGHT / (SIGMA_T * SLAB_HEIGHT_CM * state.seed_flux_model)
        field_physical = state.internal_field_model * flux_scale
        kernel = self._ensure_kernel(state)
        return kernel.pair_production_rate(field_physical, state.tau_grid)

    @staticmethod
    def pair_annihilation_rate(theta: float, tau_t: float) -> float:
        log_term = math.log(1.3 + 2.0 * EULER_ETA * theta)
        ann_shape = math.pi / (1.0 + 2.0 * theta * theta / max(log_term, 1.0e-12))
        n_species = tau_t / (2.0 * SIGMA_T * SLAB_HEIGHT_CM)
        return n_species * n_species * CLIGHT * R_E * R_E * ann_shape

    def solve_point(self, theta: float, guess_tau: float | None) -> dict[str, float | str]:
        tau_t, root_method = self.find_tau_root(theta, guess_tau)
        state = self.run_state(theta, tau_t)
        prod_coeff = self.pair_production_rate_per_ldiss2(state)
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
            "A_required": state.amplification_required,
            "l_s_over_l_diss": state.ls_over_ldiss,
            "l_c_over_l_diss": state.lc_over_ldiss,
            "pair_production_rate_unit_ldiss2": prod_coeff,
            "pair_annihilation_rate": ann_rate,
            "energy_log_residual": math.log(state.amplification_model / state.amplification_required),
            "root_method": root_method,
        }


def scan_theta_grid(config: ScanConfig) -> list[dict[str, float | str]]:
    solver = ComppsSlabSolver(config)
    rows: list[dict[str, float | str]] = []
    previous_tau: float | None = None

    for theta in logspace(config.theta_min, config.theta_max, config.n_samples):
        row = solver.solve_point(theta, previous_tau)
        rows.append(row)
        previous_tau = float(row["tau_T"])

    return rows


def write_rows(rows: list[dict[str, float | str]], config: ScanConfig) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    fieldnames = [
        "theta",
        "kTe_keV",
        "tau_T",
        "l_diss_local",
        "eta",
        "p_sc",
        "A_model",
        "A_required",
        "l_s_over_l_diss",
        "l_c_over_l_diss",
        "pair_production_rate_unit_ldiss2",
        "pair_annihilation_rate",
        "energy_log_residual",
        "root_method",
    ]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with OUTPUT_NOTES.open("w", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "Direct slab pair-line scan using patched COMPPS.",
                    "",
                    "Setup",
                    f"- theta_min = {config.theta_min}",
                    f"- theta_max = {config.theta_max}",
                    f"- n_samples = {config.n_samples}",
                    f"- kTbb = {config.tbb_kev * 1.0e3:.3f} eV",
                    f"- cold-disk albedo = {config.albedo}",
                    "- geometry = slab",
                    "- seed injection = bottom blackbody, Lambertian",
                    "- cold-disk internal dissipation d = 0",
                    "- feedback factor g = 1",
                    "- exact anisotropy forced for all scattering orders in COMPPS",
                    "",
                    "Closure",
                    "- Energy balance uses the PS96 slab relations for d=0 and g=1.",
                    "- Pair production is computed from the exported internal field I(tau,x,mu) using the exact gamma-gamma cross-section.",
                    "- Pair annihilation uses the thermal fit quoted by PS96 Appendix A27-A28.",
                    "",
                    "Caveat",
                    "- This solver uses COMPPS for the transfer step, so the internal field includes the reprocessed soft photons and Comptonized radiation produced by COMPPS. It does not add an extra reflected-hump radiation field beyond the cold-disk albedo closure already used in the energy balance.",
                ]
            )
        )


def main() -> None:
    config = ScanConfig()
    rows = scan_theta_grid(config)
    write_rows(rows, config)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
