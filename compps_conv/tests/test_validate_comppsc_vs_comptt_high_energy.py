from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_comppsc_vs_comptt_high_energy import (
    GLOBAL_CONFIG,
    KTE_VALUES,
    TAU_FACTORS,
    TAU_VALUES,
    comparison_figure_layout,
    default_compps_max_scatter,
    find_normalization_bin,
    format_tau_factor,
    heatmap_annotation_color,
    heatmap_color_limits,
    make_temperature_figure_title,
    normalize_and_compare,
    passes_acceptance,
    select_compps_max_scatter,
    snapshot_spectrum,
)


def test_find_normalization_bin_uses_upper_bin_on_exact_edge():
    edges = np.array([1.0, 10.0, 100.0])

    assert find_normalization_bin(edges, 10.0) == 1


def test_normalize_and_compare_uses_normalization_bin_and_above():
    edges = np.array([1.0, 5.0, 10.0, 20.0, 40.0])
    comp_ps = np.array([99.0, 99.0, 2.0, 1.0])
    comp_tt = np.array([1.0, 1.0, 4.0, 2.0])

    result = normalize_and_compare(edges, comp_ps, comp_tt, 10.0)

    assert result["normalization_bin"] == 2
    assert result["valid"].tolist() == [False, False, True, True]
    assert np.allclose(result["comp_ps_normalized"][2:], [1.0, 0.5])
    assert np.allclose(result["comp_tt_normalized"][2:], [1.0, 0.5])
    assert np.allclose(result["residual"][2:], 0.0)


def test_normalize_and_compare_rejects_nonpositive_normalization_flux():
    edges = np.array([1.0, 10.0, 100.0])

    with np.testing.assert_raises(ValueError):
        normalize_and_compare(edges, np.array([1.0, 0.0]), np.array([1.0, 2.0]))


def test_acceptance_includes_exact_eight_percent_boundary():
    assert passes_acceptance({"median_abs_rel": 0.08, "p95_abs_rel": 0.08})
    assert not passes_acceptance({"median_abs_rel": 0.08001, "p95_abs_rel": 0.08})
    assert not passes_acceptance({"median_abs_rel": 0.08, "p95_abs_rel": 0.08001})


def test_scientific_grid_is_fixed():
    assert KTE_VALUES == (10.0, 20.0, 51.1, 100.0, 255.5)
    assert TAU_VALUES == (0.1, 0.3, 1.0, 2.0, 3.0, 5.0)
    assert TAU_FACTORS == (0.5, 1.0, 2.0)
    assert GLOBAL_CONFIG == {
        "seed_kT_keV": 0.005,
        "compps_geometry": 1.0,
        "compps_cos_incl": 0.5,
        "comptt_approx": 1.0,
        "normalization_energy_keV": 10.0,
        "energy_min_keV": 1e-3,
        "energy_max_keV": 1e3,
        "energy_bins": 1200,
        "flux_floor": 1e-8,
        "median_limit": 0.08,
        "p95_limit": 0.08,
        "high_tau_scatter_scale": 30.0,
    }


def test_default_compps_max_scatter_matches_model_formula():
    assert default_compps_max_scatter(0.1) == 50
    assert default_compps_max_scatter(1.0) == 54
    assert default_compps_max_scatter(5.0) == 150


def test_select_compps_max_scatter_increases_large_tau_truncation():
    assert select_compps_max_scatter(0.3) == 50
    assert select_compps_max_scatter(1.0) == 54
    assert select_compps_max_scatter(2.0) == 170
    assert select_compps_max_scatter(3.0) == 320
    assert select_compps_max_scatter(5.0) == 800


def test_snapshot_spectrum_copies_mutable_model_values():
    values = np.array([1.0, 2.0, 3.0])
    snapshot = snapshot_spectrum(values)

    values[:] = 0.0

    assert np.allclose(snapshot, [1.0, 2.0, 3.0])


def test_temperature_figure_title_uses_a_real_newline():
    title = make_temperature_figure_title(20.0)

    assert "\n" in title
    assert "\\n" not in title


def test_heatmap_color_limits_cover_positive_values_for_log_scale():
    vmin, vmax = heatmap_color_limits([0.0, 0.2, 8.0, 44800.0])

    assert 0.0 < vmin <= 0.2
    assert vmax >= 44800.0


def test_comparison_figure_layout_has_one_column_per_tau_factor():
    assert comparison_figure_layout(TAU_FACTORS) == (2, 3)


def test_tau_factor_format_preserves_half_mapping():
    assert format_tau_factor(0.5) == "0.5"
    assert format_tau_factor(1.0) == "1"


def test_heatmap_annotation_color_is_readable_at_color_scale_extremes():
    assert heatmap_annotation_color(0.0) == "white"
    assert heatmap_annotation_color(0.5) == "black"
    assert heatmap_annotation_color(1.0) == "white"
