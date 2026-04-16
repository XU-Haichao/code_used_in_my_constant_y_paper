#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "inv_compton"
BUILD_DIR = ROOT / "output" / "inv_compton_build"
BIN_PATH = BUILD_DIR / "inv_compton_sim"
THERMAL_TABLE_PATH = BUILD_DIR / "thermal_kn_transport_table.h5"


def make_log_grid(min_value: float, max_value: float, count: int) -> list[float]:
    if count <= 1:
        return [min_value]
    log_min = math.log(min_value)
    log_max = math.log(max_value)
    return [
        math.exp(log_min + (log_max - log_min) * i / (count - 1))
        for i in range(count)
    ]


def build_binary() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted(str(path) for path in SRC_DIR.glob("*.cpp"))
    cmd = [
        "/usr/bin/clang++",
        "-std=c++17",
        "-O3",
        "-I",
        "/Users/epiphyllum/anaconda3/include",
        *sources,
        "-L",
        "/Users/epiphyllum/anaconda3/lib",
        "-lhdf5_cpp",
        "-lhdf5",
        "-Wl,-rpath,/Users/epiphyllum/anaconda3/lib",
        "-o",
        str(BIN_PATH),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def ensure_thermal_kn_table() -> None:
    if THERMAL_TABLE_PATH.exists():
        return
    cmd = [
        str(BIN_PATH),
        "--mode",
        "generate-thermal-kn-transport-table",
        "--output-dir",
        str(BUILD_DIR),
        "--thermal-kn-table",
        str(THERMAL_TABLE_PATH),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def read_single_row_csv(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
        if row is None:
            raise RuntimeError(f"CSV has no data rows: {path}")
        return row


def run_case(
    injection: str,
    tau: float,
    run_index: int,
    out_dir: Path,
    events: int,
    photon_energy: float,
    theta_e: float,
    max_scatters: int,
    seed_base: int,
) -> dict[str, str]:
    label = f"{injection}_tau_{run_index:03d}"
    cmd = [
        str(BIN_PATH),
        "--mode",
        "production-slab-thermal-case",
        "--events",
        str(events),
        "--seed",
        str(seed_base + run_index),
        "--photon-energy",
        f"{photon_energy:.16g}",
        "--transport-cross-section",
        "thermal_kn",
        "--thermal-kn-table",
        str(THERMAL_TABLE_PATH),
        "--electron-kTe",
        f"{theta_e:.16g}",
        "--slab-tau",
        f"{tau:.16g}",
        "--slab-injection",
        injection,
        "--max-scatters",
        str(max_scatters),
        "--label",
        label,
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    table_candidates = sorted(
        out_dir.glob(f"*{label}*_production_slab_thermal_case_table.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not table_candidates:
        raise RuntimeError(f"Could not find per-run table for label={label}")
    row = read_single_row_csv(table_candidates[-1])
    row["incident_photon_energy_mec2"] = f"{photon_energy:.16g}"
    row["events_requested"] = str(events)
    row["seed"] = str(seed_base + run_index)
    return row


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the inv_compton tau scan requested for analysis.")
    parser.add_argument("--events", type=int, default=50000)
    parser.add_argument("--tau-count", type=int, default=100)
    parser.add_argument("--tau-min", type=float, default=1.0e-2)
    parser.add_argument("--tau-max", type=float, default=1.0e1)
    parser.add_argument("--photon-energy", type=float, default=1.0e-5)
    parser.add_argument("--theta-e", type=float, default=0.1)
    parser.add_argument("--max-scatters", type=int, default=256)
    parser.add_argument("--seed-base", type=int, default=2026041201)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "inv_compton_tau_scan_eps1e-5_theta0p1_n50000_tau100",
    )
    args = parser.parse_args()

    build_binary()
    ensure_thermal_kn_table()

    tau_grid = make_log_grid(args.tau_min, args.tau_max, args.tau_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for injection in ("beam", "lambert"):
        injection_dir = args.output_dir / injection
        injection_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, str]] = []
        for index, tau in enumerate(tau_grid):
            rows.append(
                run_case(
                    injection=injection,
                    tau=tau,
                    run_index=index,
                    out_dir=injection_dir,
                    events=args.events,
                    photon_energy=args.photon_energy,
                    theta_e=args.theta_e,
                    max_scatters=args.max_scatters,
                    seed_base=args.seed_base + (0 if injection == "beam" else 100000),
                )
            )
        summary_path = args.output_dir / f"{injection}_tau_scan_summary.csv"
        write_rows(summary_path, rows)


if __name__ == "__main__":
    main()
