#!/usr/bin/env python3
"""Compute actual compPSc/ireflect values on a dense kTe-tau grid.

This scanner intentionally does not fill missing cells by interpolation.  A
matrix cell is written only when the corresponding compPSc point satisfies the
configured convergence criterion.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


ROOT_FOR_IMPORT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from pair_balance.scanner_comppsc import ComppscScanConfig, ComppscSlabSolver
from pair_balance.scanner_comppsc_eta_albedo_grid import (
    column_suffix,
    fixed_xi_column,
    row_is_converged,
)
from pair_balance.scanner_comppsc_ireflect import ev_to_kelvin
from pair_balance.scanner_reflect import IonizedReflectionConfig, IonizedReflectionKernel


MEC2_KEV = 511.0
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pair_balance" / "data"


@dataclass(frozen=True)
class ActualGridPaths:
    long_csv: pathlib.Path
    eta_matrix: pathlib.Path
    a_model_matrix: pathlib.Path
    p_sc_matrix: pathlib.Path
    albedo_matrix: pathlib.Path
    valid_matrix: pathlib.Path


def build_log_grid(minimum: float, maximum: float, n_points: int) -> np.ndarray:
    if minimum <= 0.0 or maximum <= minimum:
        raise ValueError("log grid bounds must satisfy 0 < minimum < maximum")
    if n_points < 2:
        raise ValueError("log grid needs at least two points")
    return np.geomspace(float(minimum), float(maximum), int(n_points))


def output_paths(*, n_kte: int, n_tau: int, fixed_xi: float) -> ActualGridPaths:
    slug = f"xi{column_suffix(fixed_xi)}_log{int(n_kte)}x{int(n_tau)}"
    return ActualGridPaths(
        long_csv=DATA_DIR / f"comppsc_actual_{slug}_grid.csv",
        eta_matrix=DATA_DIR / f"comppsc_actual_eta_{slug}_matrix.csv",
        a_model_matrix=DATA_DIR / f"comppsc_actual_A_model_{slug}_matrix.csv",
        p_sc_matrix=DATA_DIR / f"comppsc_actual_p_sc_{slug}_matrix.csv",
        albedo_matrix=DATA_DIR / f"comppsc_actual_albedo_ireflect_{slug}_matrix.csv",
        valid_matrix=DATA_DIR / f"comppsc_actual_valid_{slug}_matrix.csv",
    )


def format_float(value: float) -> str:
    return f"{float(value):.16g}"


def parse_float_list(values: list[str] | None) -> list[float] | None:
    if not values:
        return None
    parsed: list[float] = []
    for value in values:
        parsed.extend(float(item) for item in value.split(",") if item)
    return parsed


def point_key(kTe_kev: float, tau_t: float) -> tuple[float, float]:
    return (round(float(kTe_kev), 12), round(float(tau_t), 12))


def read_existing_rows(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def matrix_from_rows(
    rows: Sequence[dict[str, float | int | str | bool]],
    *,
    value_key: str,
    kTe_values: Sequence[float],
    tau_values: Sequence[float],
) -> np.ndarray:
    matrix = np.full((len(tau_values), len(kTe_values)), np.nan, dtype=float)
    index = {
        point_key(float(row["kTe_keV"]), float(row["tau_T"])): row
        for row in rows
    }
    for tau_index, tau_t in enumerate(tau_values):
        for kte_index, kTe_kev in enumerate(kTe_values):
            row = index.get(point_key(kTe_kev, tau_t))
            if row is None:
                continue
            if value_key == "converged":
                matrix[tau_index, kte_index] = 1.0 if row_is_converged(row) else 0.0
            elif row_is_converged(row):
                matrix[tau_index, kte_index] = float(row[value_key])
    return matrix


def write_matrix_csv(
    path: pathlib.Path,
    matrix: np.ndarray,
    *,
    kTe_values: Sequence[float],
    tau_values: Sequence[float],
    value_prefix: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tau_T", *[f"{value_prefix}_kTe_{column_suffix(kTe)}" for kTe in kTe_values]]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for tau_t, values in zip(tau_values, matrix, strict=True):
            row: list[str] = [format_float(tau_t)]
            for value in values:
                row.append("" if not math.isfinite(float(value)) else format_float(float(value)))
            writer.writerow(row)


def compute_row(
    *,
    solver: ComppscSlabSolver,
    reflector: IonizedReflectionKernel,
    kTe_kev: float,
    tau_t: float,
    fixed_xi: float,
    max_scatter: int,
    convergence_tolerance: float,
    tbb_kev: float,
    hemisphere_mu_order: int,
    observer_mu: float,
    exact_angles: bool,
) -> dict[str, float | int | bool]:
    theta = float(kTe_kev) / MEC2_KEV
    state = solver.run_state(theta, float(tau_t))
    last_scatter = int(getattr(solver.module.msiterstat, "lastisc", -1))
    last_difmax = float(getattr(solver.module.msiterstat, "lastdifmax", math.nan))
    converged = (
        last_scatter < max_scatter - 1
        and math.isfinite(last_difmax)
        and last_difmax <= convergence_tolerance
    )
    _, _, _, hemisphere_flux = reflector.hemisphere_response(
        state.x_grid,
        state.x_weights,
        state.comp_down_spectrum_model,
        observer_mu,
    )
    albedo = min(max(float(hemisphere_flux) / max(state.comp_down_flux_model, 1.0e-30), 0.0), 0.999999)
    return {
        "kTe_keV": float(kTe_kev),
        "theta": theta,
        "tau_T": float(tau_t),
        "fixed_xi": float(fixed_xi),
        "eta": state.eta,
        "p_sc": state.p_sc,
        "A_model": state.amplification_model,
        fixed_xi_column(fixed_xi): albedo,
        "downward_flux_model": state.comp_down_flux_model,
        "last_scatter_order": last_scatter,
        "last_difmax": last_difmax,
        "converged": converged,
        "max_scatter": max_scatter,
        "convergence_tolerance": convergence_tolerance,
        "tbb_keV": tbb_kev,
        "disk_temperature_K": ev_to_kelvin(1000.0 * tbb_kev),
        "hemisphere_mu_order": hemisphere_mu_order,
        "observer_mu": observer_mu,
        "exact_angles": exact_angles,
    }


def write_long_rows(path: pathlib.Path, rows: Sequence[dict[str, float | int | str | bool]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_all_matrices(
    paths: ActualGridPaths,
    *,
    rows: Sequence[dict[str, float | int | str | bool]],
    kTe_values: Sequence[float],
    tau_values: Sequence[float],
    fixed_xi: float,
) -> None:
    albedo_key = fixed_xi_column(fixed_xi)
    outputs = [
        (paths.eta_matrix, "eta", "eta"),
        (paths.a_model_matrix, "A_model", "A_model"),
        (paths.p_sc_matrix, "p_sc", "p_sc"),
        (paths.albedo_matrix, albedo_key, albedo_key),
        (paths.valid_matrix, "converged", "valid"),
    ]
    for path, value_key, prefix in outputs:
        matrix = matrix_from_rows(rows, value_key=value_key, kTe_values=kTe_values, tau_values=tau_values)
        write_matrix_csv(path, matrix, kTe_values=kTe_values, tau_values=tau_values, value_prefix=prefix)


def compute_grid(
    *,
    kTe_values: Sequence[float],
    tau_values: Sequence[float],
    fixed_xi: float,
    max_scatter: int,
    convergence_tolerance: float,
    tbb_kev: float,
    hemisphere_mu_order: int,
    observer_mu: float,
    exact_angles: bool,
    resume: bool,
) -> tuple[list[dict[str, float | int | str | bool]], ActualGridPaths]:
    paths = output_paths(n_kte=len(kTe_values), n_tau=len(tau_values), fixed_xi=fixed_xi)
    existing_rows = read_existing_rows(paths.long_csv) if resume else []
    rows: list[dict[str, float | int | str | bool]] = [dict(row) for row in existing_rows]
    done = {point_key(float(row["kTe_keV"]), float(row["tau_T"])) for row in rows}

    disk_temperature_k = ev_to_kelvin(1000.0 * tbb_kev)
    solver = ComppscSlabSolver(
        ComppscScanConfig(
            theta_min=min(kTe_values) / MEC2_KEV,
            theta_max=max(kTe_values) / MEC2_KEV,
            n_samples=1,
            tbb_kev=tbb_kev,
            max_scatter=max_scatter,
            exact_angles=exact_angles,
            observer_mu=observer_mu,
            tau_min=min(tau_values),
            tau_max=max(tau_values),
        )
    )
    reflector = IonizedReflectionKernel(
        IonizedReflectionConfig(
            disk_temperature_k=disk_temperature_k,
            ionization_parameter=float(fixed_xi),
            hemisphere_mu_order=hemisphere_mu_order,
        )
    )

    total = len(kTe_values) * len(tau_values)
    for kTe_kev in kTe_values:
        for tau_t in tau_values:
            if point_key(kTe_kev, tau_t) in done:
                continue
            row = compute_row(
                solver=solver,
                reflector=reflector,
                kTe_kev=float(kTe_kev),
                tau_t=float(tau_t),
                fixed_xi=fixed_xi,
                max_scatter=max_scatter,
                convergence_tolerance=convergence_tolerance,
                tbb_kev=tbb_kev,
                hemisphere_mu_order=hemisphere_mu_order,
                observer_mu=observer_mu,
                exact_angles=exact_angles,
            )
            rows.append(row)
            done.add(point_key(kTe_kev, tau_t))
            write_long_rows(paths.long_csv, rows)
            write_all_matrices(paths, rows=rows, kTe_values=kTe_values, tau_values=tau_values, fixed_xi=fixed_xi)
            print(
                f"[{len(done)}/{total}] "
                f"kTe={kTe_kev:g} tau={tau_t:g} "
                f"eta={float(row['eta']):.6g} A={float(row['A_model']):.6g} "
                f"p_sc={float(row['p_sc']):.6g} a={float(row[fixed_xi_column(fixed_xi)]):.6g} "
                f"sc={int(row['last_scatter_order'])} dif={float(row['last_difmax']):.3g} "
                f"conv={row['converged']}",
                flush=True,
            )

    write_long_rows(paths.long_csv, rows)
    write_all_matrices(paths, rows=rows, kTe_values=kTe_values, tau_values=tau_values, fixed_xi=fixed_xi)
    return rows, paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-kte", type=int, default=16)
    parser.add_argument("--n-tau", type=int, default=16)
    parser.add_argument("--kTe-min", type=float, default=10.0)
    parser.add_argument("--kTe-max", type=float, default=200.0)
    parser.add_argument("--tau-min", type=float, default=0.03)
    parser.add_argument("--tau-max", type=float, default=10.0)
    parser.add_argument("--kTe-grid", nargs="*", default=None, help="Optional explicit kTe values.")
    parser.add_argument("--tau-grid", nargs="*", default=None, help="Optional explicit tau values.")
    parser.add_argument("--fixed-xi", type=float, default=100.0)
    parser.add_argument("--max-scatter", type=int, default=2000)
    parser.add_argument("--convergence-tolerance", type=float, default=3.2e-3)
    parser.add_argument("--tbb-kev", type=float, default=0.005)
    parser.add_argument("--mu-order", type=int, default=8)
    parser.add_argument("--observer-mu", type=float, default=0.5)
    parser.add_argument("--no-exact-angles", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explicit_kte = parse_float_list(args.kTe_grid)
    explicit_tau = parse_float_list(args.tau_grid)
    kTe_values = explicit_kte if explicit_kte is not None else build_log_grid(args.kTe_min, args.kTe_max, args.n_kte)
    tau_values = explicit_tau if explicit_tau is not None else build_log_grid(args.tau_min, args.tau_max, args.n_tau)
    rows, paths = compute_grid(
        kTe_values=kTe_values,
        tau_values=tau_values,
        fixed_xi=args.fixed_xi,
        max_scatter=args.max_scatter,
        convergence_tolerance=args.convergence_tolerance,
        tbb_kev=args.tbb_kev,
        hemisphere_mu_order=args.mu_order,
        observer_mu=args.observer_mu,
        exact_angles=not args.no_exact_angles,
        resume=args.resume,
    )
    converged = sum(1 for row in rows if row_is_converged(row))
    print(f"rows={len(rows)} converged={converged} unconverged={len(rows) - converged}")
    print(paths.long_csv)
    print(paths.eta_matrix)
    print(paths.a_model_matrix)
    print(paths.p_sc_matrix)
    print(paths.albedo_matrix)
    print(paths.valid_matrix)


if __name__ == "__main__":
    main()
