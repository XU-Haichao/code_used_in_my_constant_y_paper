from __future__ import annotations

import json
import inspect

import numpy as np
import pandas as pd
import pytest

from pair_balance.plot_actual_parameter_maps import (
    DOUBLE_COLUMN_FIGSIZE,
    MatrixTable,
    ROOT,
    Y_RIDGE_BAND_ALPHA,
    compute_compTT_main_y_statistics,
    geometric_edges,
    geometric_edges_with_lower_bound,
    mask_unconverged,
    parameter_plot_specs,
    plot_parameter_maps,
    parse_args,
    parse_kte_header,
    restrict_tau_range,
    tau_band_from_bessel_y,
    tau_curve_from_bessel_y,
)


def test_parse_kte_header_accepts_column_suffix_notation() -> None:
    assert parse_kte_header("eta_kTe_10") == pytest.approx(10.0)
    assert parse_kte_header("eta_kTe_12p2106") == pytest.approx(12.2106)


def test_geometric_edges_extend_log_grid() -> None:
    edges = geometric_edges(np.array([10.0, 100.0, 1000.0]))

    assert edges[1] == pytest.approx(10.0**1.5)
    assert edges[2] == pytest.approx(10.0**2.5)
    assert edges[0] == pytest.approx(10.0**0.5)
    assert edges[-1] == pytest.approx(10.0**3.5)


def test_mask_unconverged_turns_invalid_cells_to_nan() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    valid = np.array([[1.0, 0.0], [1.0, 1.0]])

    masked = mask_unconverged(values, valid)

    assert masked[0, 0] == pytest.approx(1.0)
    assert np.isnan(masked[0, 1])
    assert masked[1, 1] == pytest.approx(4.0)


def test_compute_compTT_main_y_statistics_matches_notebook_selection() -> None:
    frame = pd.DataFrame(
        {
            "Source": ["A", "NGC 5506", "MCG-5-23-16", "B", "C"],
            "Electron_Temperature_keV": [50.0, 50.0, 50.0, 100.0, 150.0],
            "Optical_Depth_tau": [0.5, 0.5, 0.5, 0.25, 0.2],
            "Eddington_Ratio": [0.1, 0.1, 0.032, 0.005, 0.2],
        }
    )

    stats = compute_compTT_main_y_statistics(frame)

    assert stats.n_points == 2
    assert stats.mean_y > 0.0
    assert stats.sigma_log_y >= 0.0


def test_tau_curve_from_bessel_y_decreases_with_temperature() -> None:
    kTe_values = np.array([20.0, 100.0, 200.0])
    tau_curve = tau_curve_from_bessel_y(kTe_values, mean_y=0.8)

    assert np.all(np.isfinite(tau_curve))
    assert np.all(tau_curve > 0.0)
    assert np.all(np.diff(tau_curve) < 0.0)


def test_tau_band_from_bessel_y_encloses_mean_curve() -> None:
    kTe_values = np.array([20.0, 100.0, 200.0])
    mean_curve = tau_curve_from_bessel_y(kTe_values, mean_y=0.8)
    lower, upper = tau_band_from_bessel_y(kTe_values, mean_y=0.8, sigma_log_y=0.1)

    assert np.all(lower < mean_curve)
    assert np.all(mean_curve < upper)
    assert np.all(lower > 0.0)


def test_parameter_plot_specs_use_requested_saturated_color_limits() -> None:
    specs = {spec.key: spec for spec in parameter_plot_specs(fixed_xi=100.0, paths={})}

    assert specs["A_model"].vmin == pytest.approx(0.1)
    assert specs["A_model"].vmax == pytest.approx(500.0)
    assert specs["A_model"].extend == "both"
    assert specs["albedo"].vmin == pytest.approx(1.0e-3)
    assert specs["albedo"].vmax is None
    assert specs["albedo"].extend == "min"


def test_parameter_plot_specs_use_parameter_names_as_colorbar_labels() -> None:
    labels = {spec.key: spec.colorbar_label for spec in parameter_plot_specs(fixed_xi=100.0, paths={})}

    assert labels == {
        "eta": r"$\eta$",
        "A_model": r"$A$",
        "p_sc": r"$p_{\rm sc}$",
        "albedo": r"$a$",
    }


def test_parameter_plot_specs_use_one_colormap_for_all_panels() -> None:
    specs = parameter_plot_specs(fixed_xi=100.0, paths={})

    assert {spec.cmap for spec in specs} == {"viridis"}


def test_parameter_map_figure_uses_double_column_width() -> None:
    assert DOUBLE_COLUMN_FIGSIZE == pytest.approx((7.2, 6.0))


def test_parameter_map_plot_uses_subplot_titles_and_no_white_cross_markers() -> None:
    source = inspect.getsource(plot_parameter_maps)

    assert "fig.suptitle" not in source
    assert "        ax.set_title(spec.title)" in source
    assert "ax.scatter" not in source


def test_parameter_map_cli_defaults_to_dense_grid() -> None:
    source = inspect.getsource(parse_args)

    assert 'default="log32x32"' in source
    assert 'default="log16x16"' not in source


def test_parameter_map_uses_unlabeled_colorbars_and_second_column_y_labels() -> None:
    source = inspect.getsource(plot_parameter_maps)

    assert ".set_label(" not in source
    assert ".ax.set_title(" not in source
    assert "axes[:, 1]" in source
    assert "labelleft=True" in source


def test_notebook_contains_full_double_column_parameter_map_code() -> None:
    notebook = json.loads((ROOT / "main.ipynb").read_text())
    cell = notebook["cells"][2]
    source = "".join(cell["source"])

    assert cell["cell_type"] == "code"
    assert 'grid_tag = "log32x32"' in source
    assert "log16x16" not in source
    assert "get_double_column_size(row_height_scale=2.0)" in source
    assert "fig, axes = plt.subplots(" in source
    assert "    2,\n    2,\n" in source
    assert "fig.savefig(png_path" in source
    assert "fig.savefig(pdf_path" in source
    assert "plt.colormaps[common_cmap]" in source
    assert "fig.suptitle" not in source
    assert '    ax.set_title(spec["title"])' in source
    assert "ax.scatter" not in source
    assert "cbar.set_label" not in source
    assert "cbar.ax.set_title" not in source
    assert "axes[:, 1]" in source
    assert "labelleft=True" in source
    assert '"colorbar_label": r"$\\eta$"' in source
    assert '"colorbar_label": r"$A$"' in source
    assert '"colorbar_label": r"$p_{\\rm sc}$"' in source
    assert '"colorbar_label": r"$a$"' in source
    assert "![Actual compPSc" not in source


def test_notebook_contains_interpolated_fg_equilibrium_plot_cell() -> None:
    notebook = json.loads((ROOT / "main.ipynb").read_text())
    cell = notebook["cells"][3]
    source = "".join(cell["source"])

    assert cell["cell_type"] == "code"
    assert cell.get("metadata", {}).get("codex_added") == "fg_interpolated_equilibrium_plot"
    assert 'grid_tag = "log32x32"' in source
    assert "log16x16" not in source
    assert "f_values = (0.1, 0.3, 1.0)" in source
    assert "g_values = (0.1, 0.3, 1.0)" in source
    assert "RegularGridInterpolator" in source
    assert "brentq" in source
    assert "def energy_log_residual_grid" in source
    assert "points = np.column_stack" in source
    assert "energy_log_residual(kTe_keV, tau_T" not in source
    assert "amplification_required_fg" in source
    assert "all_points" in source
    assert "clean_points" in source
    assert "lambda_vmin, lambda_vmax = 0.01, 1.0" in source
    assert 'cbar.set_label(r"$\\lambda_{\\rm Edd}$", labelpad=-2)' in source
    assert "fig, (ax, ax_fg) = plt.subplots(" in source
    assert '"wspace": 0.24' in source
    assert "clean_compTT_points" in source
    assert "def run_fg_mcmc" in source
    assert "f_bounds = (0.9, 1.0)" in source
    assert "g_bounds = (0.001, 0.24)" in source
    assert "fg_min, fg_max" not in source
    assert "def log_observation_errors" in source
    assert "mcmc_transfer_terms" in source
    assert "def curve_distance_objective_fg" in source
    assert "def gaussian_curve_objective" in source
    assert "finite_difference_step" in source
    assert "intrinsic_log_scatter = 0.25" in source
    assert "mcmc_sigma_eff_log_kTe" in source
    assert "mcmc_sigma_eff_log_tau" in source
    assert "sigma_R2" in source
    assert "np.log(sigma_R2[valid])" in source
    assert "def log_prior_fg" in source
    assert "residual_floor" not in source
    assert "posterior_density" in source
    assert "reduced_chi2_grid" not in source
    assert "credible_density_thresholds" in source
    assert "sorted_mass = sorted_values * weights[order]" in source
    assert "posterior_cell_weights = g_widths[:, None] * f_widths[None, :]" in source
    assert "credible_contours = credible_density_thresholds(posterior_density, (0.95, 0.68), weights=posterior_cell_weights)" in source
    assert '"#4cc9f0"' in source
    assert '"#ffffff"' in source
    assert "ax_fg.clabel(" not in source
    assert "ax_fg.text(" in source
    assert "manual_label_positions_by_mass" in source
    assert "label_x, label_y = manual_label_positions_by_mass[mass]" in source
    assert "contour_label_map" in source
    assert "contour_handles" in source
    assert "contour_legend = ax_fg.legend(" in source
    assert "# contour_legend = ax_fg.legend(" not in source
    assert "minimize" not in source
    assert "def objective_fg" not in source
    assert "find_map_fg" not in source
    assert "best_result" not in source
    assert "posterior_map_row" not in source
    assert 'posterior_samples["log_posterior"].idxmax()' not in source
    assert '"log_likelihood"' in source
    assert '"log_prior"' in source
    assert '"objective"' in source
    assert '"chi2_distance"' not in source
    assert '"best_fit"' not in source
    assert "best_fit_label" not in source
    assert 'best_fit_label = "\\n".join' not in source
    assert '"best fit"' not in source
    assert "best_f_corona" not in source
    assert "best_g_feedback" not in source
    assert "best_chi2_distance" not in source
    assert "best_reduced_chi2" not in source
    assert 'marker="*"' not in source
    assert "clip_on=False" not in source
    assert "from matplotlib.ticker import NullFormatter" in source
    assert "ax_fg.yaxis.set_minor_formatter(NullFormatter())" in source
    assert 'posterior_cbar.set_label("posterior")' in source
    assert "ax_fg.set_xlim(*f_bounds)" in source
    assert "ax_fg.set_ylim(*g_bounds)" in source
    assert "ax_fg.set_xticks([0.90, 0.95, 1.00])" in source
    assert "g_major_ticks = [0.003, 0.01, 0.03, 0.1, 0.2]" in source
    assert "ax_fg.set_yticks(g_major_ticks)" in source
    assert 'ax_fg.set_yticklabels(["0.003", "0.01", "0.03", "0.1", "0.2"])' in source
    assert "ax_fg.yaxis.set_minor_locator(NullLocator())" in source
    assert 'ax_fg.set_ylabel(r"$g$", labelpad=0)' in source
    assert "ax_fg.set_title" not in source
    assert "fg_mcmc_posterior_samples.csv" in source
    assert "fg_mcmc_summary.csv" in source
    assert "sample_handles" not in source
    assert "fg_interpolated_equilibrium_curves" in source
    assert "fig.savefig(png_path" in source
    assert "fig.savefig(pdf_path" in source


def test_notebook_does_not_keep_unweighted_fg_posterior_trial_cell() -> None:
    notebook = json.loads((ROOT / "main.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "use_observation_error_weights" not in source
    assert "unweighted_local_curve_distance" not in source
    assert "common_log_distance_scale" not in source
    assert "plot_observation_errors" not in source


def test_notebook_contains_intrinsic_scatter_sensitivity_cell() -> None:
    notebook = json.loads((ROOT / "main.ipynb").read_text())
    cell = notebook["cells"][4]
    source = "".join(cell["source"])

    assert cell["cell_type"] == "code"
    assert cell.get("metadata", {}).get("codex_added") == "fg_intrinsic_scatter_sensitivity"
    assert "scatter_sensitivity_values = (0.0, 0.1, 0.5)" in source
    assert "required_previous_names" in source
    assert "Run the previous f-g posterior cell first" in source
    assert "def curve_distance_objective_fg_for_scatter" in source
    assert "gaussian_curve_objective" in source
    assert "deterministic_posterior_grid_with_log_sigma" in source
    assert "def log_posterior_fg_for_scatter" in source
    assert "def posterior_density_grid_for_scatter" in source
    assert "f_axis_sensitivity = np.linspace(f_bounds[0], f_bounds[1], 170)" in source
    assert "g_axis_sensitivity = np.geomspace(g_bounds[0], g_bounds[1], 170)" in source
    assert "credible_density_thresholds" in source
    assert "posterior_cell_weights_sensitivity = g_widths[:, None] * f_widths[None, :]" in source
    assert "credible_contours = credible_density_thresholds(density, (0.95, 0.68), weights=posterior_cell_weights_sensitivity)" in source
    assert "fg_intrinsic_scatter_sensitivity.png" in source
    assert "fg_intrinsic_scatter_sensitivity.pdf" in source
    assert "fg_intrinsic_scatter_sensitivity_summary.csv" in source
    assert "fg_intrinsic_scatter_sensitivity_density.csv" in source
    assert "posterior_density_sensitivity" in source


def test_notebook_contains_asymmetric_error_posterior_comparison_cell() -> None:
    notebook = json.loads((ROOT / "main.ipynb").read_text())
    matching_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("codex_added") == "fg_asymmetric_error_posterior_comparison"
    ]

    assert len(matching_cells) == 1
    source = "".join(matching_cells[0]["source"])
    assert "error_scatter_values = (0.0, 0.25)" in source
    assert 'error_model_values = ("mean", "asymmetric")' in source
    assert "def log_observation_error_components" in source
    assert "direction_log_kTe = -residual * dR_dlog_kTe" in source
    assert "direction_log_tau = -residual * dR_dlog_tau" in source
    assert "def curve_distance_objective_fg_for_error_model" in source
    assert "np.log(sigma_R2[valid])" in source
    assert "def posterior_density_grid_for_error_model" in source
    assert "posterior_cell_weights_error_test = g_widths_error_test[:, None] * f_widths_error_test[None, :]" in source
    assert "credible_contours = credible_density_thresholds(density, (0.95, 0.68), weights=posterior_cell_weights_error_test)" in source
    assert "fg_asymmetric_error_posterior_comparison.png" in source
    assert "fg_asymmetric_error_posterior_comparison.pdf" in source
    assert "fg_asymmetric_error_posterior_comparison_summary.csv" in source
    assert "fg_asymmetric_error_posterior_comparison_density.csv" in source
    assert "fig.savefig(error_test_png" in source
    assert "fig.savefig(error_test_pdf" in source


def test_notebook_removed_temporary_likelihood_normalization_comparison_cell() -> None:
    notebook = json.loads((ROOT / "main.ipynb").read_text())
    matching_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("codex_added") == "fg_likelihood_normalization_comparison"
    ]

    assert matching_cells == []
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "fg_likelihood_normalization_comparison" not in source
    assert 'normalization_likelihood_values = ("chi2_only", "with_log_sigma")' not in source


def test_notebook_contains_comptt_spearman_correlation_cell() -> None:
    notebook = json.loads((ROOT / "main.ipynb").read_text())
    matching_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("codex_added") == "compTT_spearman_correlations"
    ]

    assert len(matching_cells) == 1
    assert matching_cells[0] is notebook["cells"][-1]
    source = "".join(matching_cells[0]["source"])
    assert "compTT_comptt_sample_v2.csv" in source
    assert "Optical_Depth_tau" in source
    assert "frame[col] *= 2.0" in source
    assert "NGC 5506" in source
    assert "use_clean_sample" in source
    assert "bessel_factor" in source
    assert "spearmanr" in source
    assert "cleaned compTT" in source
    assert "full compTT" in source
    assert "compTT_spearman_correlations.csv" in source


def test_restrict_tau_range_drops_rows_below_minimum() -> None:
    table = MatrixTable(
        tau_values=np.array([0.03, 0.1, 0.3]),
        kTe_values=np.array([10.0, 20.0]),
        values=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )

    restricted = restrict_tau_range(table, tau_min=0.1)

    assert restricted.tau_values.tolist() == [0.1, 0.3]
    assert restricted.values.tolist() == [[3.0, 4.0], [5.0, 6.0]]


def test_geometric_edges_with_lower_bound_starts_first_cell_at_tau_min() -> None:
    edges = geometric_edges_with_lower_bound(np.array([0.208, 0.306, 0.451]), lower_bound=0.2)

    assert edges[0] == pytest.approx(0.2)
    assert edges[1] > 0.2


def test_y_ridge_band_alpha_is_visible_but_not_opaque() -> None:
    assert 0.1 <= Y_RIDGE_BAND_ALPHA <= 0.3


def test_notebook_saves_all_generated_figures_under_figure_directory() -> None:
    notebook = json.loads((ROOT / "main.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert 'figure_dir = Path("figure")' in source
    assert "png_path = output_dir" not in source
    assert "pdf_path = output_dir" not in source
    assert "sensitivity_png = output_dir" not in source
    assert "sensitivity_pdf = output_dir" not in source


def test_notebook_plot_cells_keep_grid_disabled() -> None:
    notebook = json.loads((ROOT / "main.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert ".grid(True" not in source
    assert '"axes.grid": True' not in source


def test_notebook_pair_balance_cell_uses_recomputed_comppsc_fg_curves() -> None:
    notebook = json.loads((ROOT / "main.ipynb").read_text())
    cell = notebook["cells"][6]
    source = "".join(cell["source"])

    assert cell["cell_type"] == "code"
    assert "comppsc_fg_pair_balance_xi100_f0p1_0p3_1_g0p1_0p3_1_maxsc4000.csv" in source
    assert "ps96_slab_dratio" not in source
    assert 'groupby(["f_corona", "g_feedback"]' in source
    assert "g_linestyles" in source


def test_notebook_contains_sample_compactness_pair_balance_cell() -> None:
    notebook = json.loads((ROOT / "main.ipynb").read_text())
    matching_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("codex_added") == "sample_compactness_pair_balance"
    ]

    assert len(matching_cells) == 1
    source = "".join(matching_cells[0]["source"])
    assert "comppsc_fg_pair_balance_xi100_f0p1_0p3_1_g0p1_0p3_1_maxsc4000.csv" in source
    assert "compactness_prefactor = 2.3e4" in source
    assert "f_compactness = 1.0" in source
    assert "h_over_rg_values = (1.0, 10.0)" in source
    assert "marker_map = {\"compTT\": \"s\", \"compPS\": \"^\"}" in source
    assert "f_values = tuple(sorted(pair_curves[\"f_corona\"].dropna().unique()))" in source
    assert "g_values = tuple(sorted(pair_curves[\"g_feedback\"].dropna().unique()))" in source
    assert "f_colors" in source
    assert "g_linestyles" in source
    assert "sample_ldiss_low" in source
    assert "sample_ldiss_high" in source
    assert "No Pair" in source
    assert "No Pair\\nBalance" in source
    assert 'ax_kte.text(0.96, 0.52, "No Pair\\nBalance"' in source
    assert 'ax_tau.text(0.06, 0.52, "No Pair\\nBalance"' in source
    assert 'bbox=no_pair_balance_bbox' in source
    assert 'zorder=20' in source
    assert '"boxstyle": "round,pad=0.20"' in source
    assert '"facecolor": "#ffffff"' in source
    assert '"edgecolor": "0.72"' in source
    assert '"linewidth": 0.0' in source
    assert '"alpha": 0.94' in source
    assert "sample_handles =" not in source
    assert 'label="compTT"' not in source
    assert 'label="compPS"' not in source
    assert "pair_balance_samples_fg_grid" in source
    assert "figure_dir" in source
    assert ".grid(True" not in source
