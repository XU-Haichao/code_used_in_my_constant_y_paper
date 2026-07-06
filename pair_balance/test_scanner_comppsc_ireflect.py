from __future__ import annotations

import math

import numpy as np
import pytest

from pair_balance.scanner_reflect import IonizedReflectionConfig, IonizedReflectionKernel
from pair_balance.scanner_comppsc_ireflect import (
    IreflectComppscConfig,
    IreflectComppscSolver,
    band_energy_flux,
    blackbody_surface_flux,
    ev_to_kelvin,
    ionization_parameter,
)


def test_five_ev_disk_temperature_is_in_kelvin() -> None:
    assert ev_to_kelvin(5.0) == pytest.approx(58022.5906, rel=1.0e-8)


def test_blackbody_surface_flux_uses_stefan_boltzmann_law() -> None:
    temperature_k = ev_to_kelvin(5.0)
    expected = 5.670374419e-5 * temperature_k**4

    assert blackbody_surface_flux(temperature_k) == pytest.approx(expected, rel=1.0e-13)


def test_band_energy_flux_conserves_full_native_grid_integral() -> None:
    energy_kev = np.logspace(-3.0, 2.0, 8)
    x_grid = energy_kev / 511.0
    x_weights = np.full(energy_kev.size, math.log(energy_kev[1] / energy_kev[0]))
    spectrum = np.arange(1.0, energy_kev.size + 1.0)
    expected = float(np.sum(x_grid * x_weights * spectrum))

    measured = band_energy_flux(
        x_grid,
        x_weights,
        spectrum,
        energy_min_kev=0.0,
        energy_max_kev=math.inf,
    )

    assert measured == pytest.approx(expected, rel=1.0e-13)


def test_ionization_parameter_scales_inversely_with_density() -> None:
    f_ion = 8.0e14
    xi_13 = ionization_parameter(f_ion, density_cm3=1.0e13)
    xi_15 = ionization_parameter(f_ion, density_cm3=1.0e15)

    assert xi_13 == pytest.approx(4.0 * math.pi * f_ion / 1.0e13)
    assert xi_13 / xi_15 == pytest.approx(100.0)


def test_ireflect_bridge_returns_nonnegative_hemisphere_spectrum() -> None:
    x_grid = np.logspace(-5.0, math.log10(2.0), 100)
    x_weights = np.full(x_grid.size, math.log(x_grid[1] / x_grid[0]))
    incident = x_grid ** -0.8 * np.exp(-x_grid / 0.1)
    kernel = IonizedReflectionKernel(
        IonizedReflectionConfig(
            disk_temperature_k=ev_to_kelvin(5.0),
            ionization_parameter=10.0,
            hemisphere_mu_order=4,
        )
    )

    observer, hemisphere, observer_flux, hemisphere_flux = kernel.hemisphere_response(
        x_grid,
        x_weights,
        incident,
        observer_mu=0.5,
    )

    assert np.all(np.isfinite(observer))
    assert np.all(np.isfinite(hemisphere))
    assert np.all(observer >= 0.0)
    assert np.all(hemisphere >= 0.0)
    assert observer_flux > 0.0
    assert hemisphere_flux > 0.0


def test_one_pass_ireflect_state_has_physical_energy_terms() -> None:
    solver = IreflectComppscSolver(
        IreflectComppscConfig(
            kTe_kev=10.0,
            density_cm3=1.0e15,
            max_scatter=300,
            hemisphere_mu_order=4,
        )
    )

    result = solver.evaluate_tau(2.5)

    assert result["ionizing_flux_cgs"] > 0.0
    assert result["ionization_parameter"] > 0.0
    assert 0.0 <= result["effective_albedo"] < 1.0
    assert result["A_model"] > 0.0
    assert result["A_required"] > 0.0
    assert math.isfinite(result["energy_log_residual"])
    assert result["last_scatter_order"] < 300
    assert result["last_difmax"] <= 3.1e-3


def test_one_pass_ireflect_solver_finds_energy_balance_tau() -> None:
    solver = IreflectComppscSolver(
        IreflectComppscConfig(
            kTe_kev=10.0,
            density_cm3=1.0e15,
            max_scatter=300,
            hemisphere_mu_order=4,
            tau_min=2.0,
            tau_max=3.0,
            root_tolerance=8.0e-4,
        )
    )

    result = solver.solve_tau()

    assert 2.0 < result["tau_T"] < 3.0
    assert abs(result["energy_log_residual"]) < 8.0e-4
