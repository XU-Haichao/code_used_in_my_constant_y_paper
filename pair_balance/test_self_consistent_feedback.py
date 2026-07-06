from __future__ import annotations

import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from pair_balance.scanner_reflect import resolve_heasoft_root
from pair_balance.self_consistent_feedback import (
    FeedbackGeneration,
    FullFeedbackComppscSolver,
    FullFeedbackConfig,
    bisect_feedback_root,
    blackbody_seed_bins,
    disk_response,
    energy_flux,
    native_energy_edges,
    normalize_seed_energy,
    power_iteration,
    run_impulse_response,
    spectrum_to_seed_bins,
)


def test_spectrum_to_seed_bins_preserves_native_grid_energy() -> None:
    x_grid = np.array([1.0e-3, 2.0e-3, 4.0e-3])
    x_weights = np.array([0.4, 0.7, 0.4])
    energy_edges_kev = np.array([0.35, 0.72, 1.45, 2.90])
    spectrum = np.array([2.0, 3.0, 5.0])

    seed_bins = spectrum_to_seed_bins(x_grid, x_weights, energy_edges_kev, spectrum)
    seed_energy = float(np.dot(np.sqrt(energy_edges_kev[:-1] * energy_edges_kev[1:]) / 511.0, seed_bins))

    assert seed_energy == pytest.approx(energy_flux(x_grid, x_weights, spectrum), rel=1.0e-13)
    assert np.all(seed_bins >= 0.0)


def test_disk_response_conserves_downward_energy() -> None:
    x_grid = np.array([1.0e-3, 2.0e-3, 4.0e-3])
    x_weights = np.array([0.4, 0.7, 0.4])
    energy_edges_kev = np.array([0.35, 0.72, 1.45, 2.90])
    downward = np.array([2.0, 3.0, 5.0])
    reflected = 0.25 * downward
    blackbody_bins = normalize_seed_energy(
        energy_edges_kev,
        np.array([5.0, 2.0, 1.0]),
        target_energy=1.0,
    )

    response = disk_response(
        x_grid,
        x_weights,
        energy_edges_kev,
        downward,
        reflected,
        blackbody_bins,
    )
    weights = np.sqrt(energy_edges_kev[:-1] * energy_edges_kev[1:]) / 511.0

    assert response.reflected_energy == pytest.approx(0.25 * response.downward_energy)
    assert response.absorbed_energy == pytest.approx(0.75 * response.downward_energy)
    assert float(np.dot(weights, response.returned_seed_bins)) == pytest.approx(response.downward_energy)
    assert response.energy_residual == pytest.approx(0.0, abs=1.0e-13)


def test_power_iteration_finds_asymptotic_return_gain() -> None:
    energy_weights = np.array([1.0, 2.0])
    initial_seed = np.array([1.0, 0.0])

    def fake_round_trip(seed_bins: np.ndarray, tau_t: float) -> FeedbackGeneration:
        returned = np.array([0.25, 0.375]) * tau_t
        return FeedbackGeneration(
            returned_seed_bins=returned,
            input_energy=float(np.dot(energy_weights, seed_bins)),
            unscattered_up_energy=0.1,
            compton_up_energy=0.2,
            compton_down_energy=float(np.dot(energy_weights, returned)),
            reflected_energy=0.0,
            absorbed_energy=float(np.dot(energy_weights, returned)),
            coronal_gain_energy=0.0,
            disk_energy_residual=0.0,
            last_scatter_order=1,
            last_difmax=0.0,
        )

    result = power_iteration(
        fake_round_trip,
        tau_t=1.5,
        initial_seed_bins=initial_seed,
        energy_weights=energy_weights,
        tolerance=1.0e-12,
        max_iterations=10,
    )

    assert result.converged
    assert result.lambda_energy == pytest.approx(1.5)
    assert np.dot(energy_weights, result.seed_shape) == pytest.approx(1.0)


def test_log_bisection_recovers_unit_feedback_root() -> None:
    root, residual, iterations = bisect_feedback_root(
        lambda tau_t: 0.5 * tau_t,
        tau_lo=0.5,
        tau_hi=5.0,
        relative_tolerance=1.0e-10,
        max_iterations=80,
    )

    assert root == pytest.approx(2.0, rel=1.0e-9)
    assert abs(residual) < 1.0e-9
    assert iterations < 80


def test_log_bisection_rejects_unbracketed_root() -> None:
    with pytest.raises(ValueError, match="bracket"):
        bisect_feedback_root(
            lambda tau_t: 0.2 * tau_t,
            tau_lo=0.5,
            tau_hi=2.0,
        )


def test_native_energy_edges_recover_logarithmic_centers() -> None:
    x_grid = np.logspace(-6.0, -3.0, 4)
    edges = native_energy_edges(x_grid)

    assert edges.size == x_grid.size + 1
    assert np.sqrt(edges[:-1] * edges[1:]) == pytest.approx(511.0 * x_grid)


def test_blackbody_seed_bins_have_requested_energy() -> None:
    edges = np.logspace(-5.0, 2.0, 101)
    seed = blackbody_seed_bins(edges, tbb_kev=0.005, target_energy=2.5)

    assert np.dot(seed_energy_weights_for_test(edges), seed) == pytest.approx(2.5)
    assert np.all(seed >= 0.0)


def seed_energy_weights_for_test(edges: np.ndarray) -> np.ndarray:
    return np.sqrt(edges[:-1] * edges[1:]) / 511.0


def test_impulse_response_accumulates_a_decaying_feedback_series() -> None:
    energy_weights = np.array([1.0])

    def fake_round_trip(seed_bins: np.ndarray, tau_t: float) -> FeedbackGeneration:
        input_energy = float(np.dot(energy_weights, seed_bins))
        returned_energy = 0.5 * input_energy
        return FeedbackGeneration(
            returned_seed_bins=np.array([returned_energy]),
            input_energy=input_energy,
            unscattered_up_energy=0.2 * input_energy,
            compton_up_energy=0.3 * input_energy,
            compton_down_energy=returned_energy,
            reflected_energy=0.1 * input_energy,
            absorbed_energy=0.4 * input_energy,
            coronal_gain_energy=0.0,
            disk_energy_residual=0.0,
            last_scatter_order=1,
            last_difmax=0.0,
        )

    result = run_impulse_response(
        fake_round_trip,
        tau_t=1.0,
        initial_seed_bins=np.array([1.0]),
        energy_weights=energy_weights,
        stop_energy=1.0e-8,
        max_generations=100,
    )

    assert result.converged
    assert result.total_input_energy == pytest.approx(2.0, rel=1.0e-7)
    assert result.total_upward_escape_energy == pytest.approx(1.0, rel=1.0e-7)
    assert result.total_absorbed_energy == pytest.approx(0.8, rel=1.0e-7)


def test_reflect_discovers_heasoft_full_environment() -> None:
    root = resolve_heasoft_root()

    assert root.name == "heasoft"
    assert (root / "lib" / "libXSFunctions.dylib").exists()


def test_real_comppsc_reflect_round_trip_conserves_disk_energy() -> None:
    solver = FullFeedbackComppscSolver(
        FullFeedbackConfig(
            kTe_kev=10.0,
            max_scatter=300,
            hemisphere_mu_order=4,
        )
    )

    generation = solver.round_trip(solver.initial_blackbody_bins, tau_t=2.0)
    returned_energy = float(np.dot(solver.energy_weights, generation.returned_seed_bins))

    assert generation.input_energy == pytest.approx(1.0)
    assert generation.compton_down_energy > 0.0
    assert returned_energy == pytest.approx(generation.compton_down_energy, rel=1.0e-8)
    assert generation.reflected_energy >= 0.0
    assert generation.absorbed_energy >= 0.0
    assert abs(generation.disk_energy_residual) < 1.0e-8
    assert generation.last_scatter_order < 300
    assert generation.last_difmax <= 3.1e-3


def test_feedback_script_runs_directly_from_project_root() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "pair_balance" / "self_consistent_feedback.py"),
            "--mode",
            "point",
            "--tau",
            "2",
            "--max-scatter",
            "300",
            "--mu-order",
            "4",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "disk_residual=" in completed.stdout
