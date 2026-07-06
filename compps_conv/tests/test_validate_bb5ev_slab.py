from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_bb5ev_slab import (
    GLOBAL_CONFIG,
    VALIDATION_CASES,
    compare_spectra,
    passes_acceptance,
)


def test_compare_spectra_removes_constant_normalization():
    reference = np.array([1.0, 2.0, 4.0, 8.0])
    candidate = 2.5 * reference

    result = compare_spectra(reference, candidate)

    assert np.isclose(result["scale"], 0.4)
    assert np.allclose(result["residual"][result["valid"]], 0.0)
    assert np.isclose(result["median_abs_rel"], 0.0)
    assert np.isclose(result["p95_abs_rel"], 0.0)


def test_compare_spectra_excludes_nonfinite_and_low_flux_bins():
    reference = np.array([1.0, 1e-12, np.nan, 0.5, 0.0])
    candidate = np.array([2.0, 2e-12, 1.0, 1.0, 1.0])

    result = compare_spectra(reference, candidate, flux_floor=1e-8)

    assert result["valid"].tolist() == [True, False, False, True, False]
    assert result["n_valid"] == 2


def test_passes_acceptance_uses_strict_agreed_limits():
    assert passes_acceptance({"median_abs_rel": 0.009, "p95_abs_rel": 0.029})
    assert not passes_acceptance({"median_abs_rel": 0.01, "p95_abs_rel": 0.029})
    assert not passes_acceptance({"median_abs_rel": 0.009, "p95_abs_rel": 0.03})


def test_validation_grid_and_shared_physics_are_fixed():
    assert VALIDATION_CASES == (
        (51.1, 0.1),
        (51.1, 1.0),
        (255.5, 0.1),
        (255.5, 1.0),
    )
    assert GLOBAL_CONFIG == {
        "seed_kT_keV": 0.005,
        "geometry": 1.0,
        "cos_incl": 0.5,
        "reflection": 0.0,
        "energy_min_keV": 1e-3,
        "energy_max_keV": 1e3,
        "energy_bins": 1200,
        "comparison_min_keV": 2e-3,
    }
