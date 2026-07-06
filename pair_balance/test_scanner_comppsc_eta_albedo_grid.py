from __future__ import annotations

import pytest

from pair_balance.scanner_comppsc_eta_albedo_grid import (
    fixed_xi_column,
    geometric_edges,
    rows_to_converged_matrix,
    rows_to_matrix,
    summarize_eta_dependence,
)


def test_geometric_edges_are_log_midpoints() -> None:
    edges = geometric_edges([1.0, 10.0, 100.0])

    assert edges[1] == pytest.approx(10.0**0.5)
    assert edges[2] == pytest.approx(10.0**1.5)
    assert edges[0] == pytest.approx(10.0**-0.5)
    assert edges[-1] == pytest.approx(10.0**2.5)


def test_rows_to_matrix_maps_kte_and_tau_axes() -> None:
    rows = [
        {"kTe_keV": 10.0, "tau_T": 0.1, "eta": 0.4},
        {"kTe_keV": 20.0, "tau_T": 0.1, "eta": 0.5},
        {"kTe_keV": 10.0, "tau_T": 1.0, "eta": 0.6},
        {"kTe_keV": 20.0, "tau_T": 1.0, "eta": 0.7},
    ]

    matrix = rows_to_matrix(rows, value_key="eta", kTe_values=[10.0, 20.0], tau_values=[0.1, 1.0])

    assert matrix.tolist() == [[0.4, 0.5], [0.6, 0.7]]


def test_fixed_xi_column_is_stable_for_csv_and_matrix_names() -> None:
    assert fixed_xi_column(100.0) == "effective_albedo_ireflect_xi100"
    assert fixed_xi_column(31.6) == "effective_albedo_ireflect_xi31p6"


def test_rows_to_matrix_can_extract_fixed_xi_albedo() -> None:
    rows = [
        {"kTe_keV": 10.0, "tau_T": 0.1, "effective_albedo_ireflect_xi100": 0.01},
        {"kTe_keV": 20.0, "tau_T": 0.1, "effective_albedo_ireflect_xi100": 0.02},
        {"kTe_keV": 10.0, "tau_T": 1.0, "effective_albedo_ireflect_xi100": 0.10},
        {"kTe_keV": 20.0, "tau_T": 1.0, "effective_albedo_ireflect_xi100": 0.20},
    ]

    matrix = rows_to_matrix(
        rows,
        value_key=fixed_xi_column(100.0),
        kTe_values=[10.0, 20.0],
        tau_values=[0.1, 1.0],
    )

    assert matrix.tolist() == [[0.01, 0.02], [0.10, 0.20]]


def test_rows_to_converged_matrix_masks_unconverged_cells() -> None:
    rows = [
        {"kTe_keV": 10.0, "tau_T": 0.1, "A_model": 1.1, "converged": True},
        {"kTe_keV": 20.0, "tau_T": 0.1, "A_model": 1.2, "converged": "False"},
        {"kTe_keV": 10.0, "tau_T": 1.0, "A_model": 2.1, "converged": True},
        {"kTe_keV": 20.0, "tau_T": 1.0, "A_model": 2.2, "converged": True},
    ]

    matrix = rows_to_converged_matrix(
        rows,
        value_key="A_model",
        kTe_values=[10.0, 20.0],
        tau_values=[0.1, 1.0],
    )

    assert matrix[0, 0] == pytest.approx(1.1)
    assert matrix[0, 1] != matrix[0, 1]
    assert matrix[1, 0] == pytest.approx(2.1)
    assert matrix[1, 1] == pytest.approx(2.2)


def test_summarize_eta_dependence_uses_converged_rows_only() -> None:
    rows = [
        {"tau_T": 1.0, "eta": 0.50, "converged": True},
        {"tau_T": 1.0, "eta": 0.55, "converged": True},
        {"tau_T": 1.0, "eta": 0.90, "converged": False},
    ]

    summary = summarize_eta_dependence(rows, [1.0])

    assert summary[0]["n_converged_kTe"] == 2
    assert summary[0]["eta_min_over_kTe"] == pytest.approx(0.50)
    assert summary[0]["eta_max_over_kTe"] == pytest.approx(0.55)
