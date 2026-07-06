from __future__ import annotations

import csv

import numpy as np
import pytest

from pair_balance.compute_comppsc_actual_dense_grid import (
    build_log_grid,
    matrix_from_rows,
    output_paths,
    write_matrix_csv,
)


def test_build_log_grid_includes_endpoints() -> None:
    grid = build_log_grid(10.0, 200.0, 5)

    assert grid[0] == pytest.approx(10.0)
    assert grid[-1] == pytest.approx(200.0)
    assert np.all(np.diff(grid) > 0.0)


def test_output_paths_are_tagged_by_grid_and_fixed_xi() -> None:
    paths = output_paths(n_kte=16, n_tau=20, fixed_xi=100.0)

    assert paths.long_csv.name == "comppsc_actual_xi100_log16x20_grid.csv"
    assert paths.eta_matrix.name == "comppsc_actual_eta_xi100_log16x20_matrix.csv"
    assert paths.valid_matrix.name == "comppsc_actual_valid_xi100_log16x20_matrix.csv"


def test_matrix_from_rows_masks_unconverged_cells() -> None:
    rows = [
        {"kTe_keV": 10.0, "tau_T": 0.1, "eta": 0.4, "converged": True},
        {"kTe_keV": 20.0, "tau_T": 0.1, "eta": 0.5, "converged": False},
        {"kTe_keV": 10.0, "tau_T": 1.0, "eta": 0.6, "converged": True},
        {"kTe_keV": 20.0, "tau_T": 1.0, "eta": 0.7, "converged": True},
    ]

    matrix = matrix_from_rows(rows, value_key="eta", kTe_values=[10.0, 20.0], tau_values=[0.1, 1.0])

    assert matrix[0, 0] == pytest.approx(0.4)
    assert np.isnan(matrix[0, 1])
    assert matrix[1, 0] == pytest.approx(0.6)
    assert matrix[1, 1] == pytest.approx(0.7)


def test_write_matrix_csv_preserves_empty_missing_values(tmp_path) -> None:
    path = tmp_path / "matrix.csv"
    matrix = np.array([[1.0, np.nan], [2.0, 3.0]])

    write_matrix_csv(path, matrix, kTe_values=[10.0, 20.0], tau_values=[0.1, 1.0], value_prefix="eta")

    rows = list(csv.reader(path.open()))
    assert rows[0] == ["tau_T", "eta_kTe_10", "eta_kTe_20"]
    assert rows[1] == ["0.1", "1", ""]
    assert rows[2] == ["1", "2", "3"]
