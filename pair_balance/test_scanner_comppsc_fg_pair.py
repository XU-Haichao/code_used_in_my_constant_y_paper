from __future__ import annotations

import csv
import math

import pytest

from pair_balance.scanner_comppsc_fg import compactness_terms_fg
from pair_balance.scanner_comppsc_fg_pair import (
    FGPairBalanceConfig,
    downsample_equilibrium_curve_rows,
    field_flux_scale_per_ldiss,
    kTe_values_for_scan,
    output_paths,
    pair_balance_ldiss,
    plot_pair_curves,
    write_rows,
)


def test_fg_pair_balance_defaults_use_requested_parameter_grid() -> None:
    config = FGPairBalanceConfig()

    assert config.f_values == (0.1, 0.3, 1.0)
    assert config.g_values == (0.1, 0.3, 1.0)
    assert config.kTe_min_kev == pytest.approx(10.0)
    assert config.kTe_max_kev == pytest.approx(200.0)
    assert config.tau_max >= 10.0
    assert config.max_scatter >= 3000


def test_output_paths_record_fg_grid_and_comppsc_transfer() -> None:
    csv_path, png_path, pdf_path = output_paths((0.1, 0.3, 1.0), (0.1, 0.3, 1.0), 100.0, 4000)

    assert csv_path.name.startswith("comppsc_fg_pair_balance_xi100_")
    assert "f0p1_0p3_1" in csv_path.name
    assert "g0p1_0p3_1" in csv_path.name
    assert "maxsc4000" in csv_path.name
    assert png_path.parent.name == "figure"
    assert pdf_path.parent.name == "figure"
    assert png_path.suffix == ".png"
    assert pdf_path.suffix == ".pdf"


def test_single_point_scan_uses_kte_min_without_logspace_division() -> None:
    assert kTe_values_for_scan(20.0, 20.0, 1) == [20.0]


def test_pair_balance_uses_fg_seed_compactness_scaling() -> None:
    eta = 0.55
    p_sc = 0.82
    albedo = 0.2
    f_corona = 0.3
    feedback_factor = 0.1

    terms = compactness_terms_fg(eta, p_sc, albedo, f_corona, feedback_factor)
    scale = field_flux_scale_per_ldiss(
        ls_over_ldiss=terms.l_s_over_l_c,
        seed_flux_model=2.0,
    )

    assert scale > 0.0
    assert scale == pytest.approx(terms.l_s_over_l_c * field_flux_scale_per_ldiss(1.0, 2.0))


def test_pair_balance_ldiss_solves_annihilation_over_production() -> None:
    assert pair_balance_ldiss(pair_production_rate_unit_ldiss2=2.0, pair_annihilation_rate=18.0) == pytest.approx(3.0)
    assert math.isinf(pair_balance_ldiss(pair_production_rate_unit_ldiss2=0.0, pair_annihilation_rate=18.0))


def test_write_rows_preserves_fg_pair_columns(tmp_path) -> None:
    path = tmp_path / "fg_pair.csv"
    rows = [
        {
            "model": "compPSc-fg",
            "f_corona": 0.1,
            "g_feedback": 0.3,
            "theta": 0.05,
            "kTe_keV": 25.55,
            "tau_T": 2.0,
            "l_diss_local": 1.0e3,
            "pair_production_rate_unit_ldiss2": 4.0,
            "pair_annihilation_rate": 4.0e6,
            "last_scatter_order": 350,
            "last_difmax": 1.0e-4,
            "converged": True,
            "max_scatter": 4000,
        }
    ]

    write_rows(path, rows)

    loaded = list(csv.DictReader(path.open()))
    assert loaded[0]["model"] == "compPSc-fg"
    assert loaded[0]["f_corona"] == "0.1"
    assert loaded[0]["g_feedback"] == "0.3"
    assert loaded[0]["l_diss_local"] == "1000.0"


def test_downsample_equilibrium_curve_rows_preserves_curve_endpoints() -> None:
    rows = [{"kTe_keV": value, "tau_T": 1.0 / value} for value in range(1, 7)]

    selected = downsample_equilibrium_curve_rows(rows, max_points=4)

    assert [row["kTe_keV"] for row in selected] == [1, 3, 4, 6]


def test_plot_pair_curves_accepts_non_default_feedback_values(tmp_path) -> None:
    rows = [
        {
            "f_corona": 1.0,
            "g_feedback": 0.05,
            "kTe_keV": 20.0,
            "tau_T": 2.0,
            "l_diss_local": 1.0e4,
        },
        {
            "f_corona": 1.0,
            "g_feedback": 0.05,
            "kTe_keV": 100.0,
            "tau_T": 0.6,
            "l_diss_local": 30.0,
        },
    ]

    png_path = tmp_path / "pair.png"
    pdf_path = tmp_path / "pair.pdf"
    plot_pair_curves(png_path, pdf_path, rows, f_values=(1.0,), g_values=(0.05,))

    assert png_path.exists()
    assert pdf_path.exists()
