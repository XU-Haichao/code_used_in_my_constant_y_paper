from __future__ import annotations

import pytest

from pair_balance.scanner_comppsc_fg import (
    amplification_required_fg,
    compactness_terms_fg,
    d_ratio_from_f,
    f_from_d_ratio,
)


def test_d_ratio_and_f_are_inverse_parameterizations() -> None:
    assert d_ratio_from_f(1.0) == pytest.approx(0.0)
    assert d_ratio_from_f(0.8) == pytest.approx(0.25)
    assert f_from_d_ratio(0.25) == pytest.approx(0.8)


def test_feedback_factor_one_recovers_old_d_ratio_closure() -> None:
    eta = 0.62
    p_sc = 0.91
    albedo = 0.18
    f_corona = 0.8
    d_ratio = d_ratio_from_f(f_corona)
    old_required = (1.0 + d_ratio * p_sc) / (
        (1.0 - albedo) * eta + d_ratio * (1.0 - albedo * eta * p_sc)
    )

    assert amplification_required_fg(eta, p_sc, albedo, f_corona, 1.0) == pytest.approx(old_required)


def test_reduced_feedback_increases_required_amplification() -> None:
    eta = 0.62
    p_sc = 0.91
    albedo = 0.18
    f_corona = 0.8

    full_covering = amplification_required_fg(eta, p_sc, albedo, f_corona, 1.0)
    patchy = amplification_required_fg(eta, p_sc, albedo, f_corona, 0.5)

    assert patchy > full_covering


def test_compactness_terms_reduce_to_old_g_equal_one_terms() -> None:
    eta = 0.55
    p_sc = 0.8
    albedo = 0.2
    f_corona = 0.5
    d_ratio = d_ratio_from_f(f_corona)
    expected_lh = (1.0 + d_ratio * p_sc) / (1.0 - p_sc * eta)
    expected_ls = d_ratio + (1.0 - albedo) * eta * expected_lh

    terms = compactness_terms_fg(eta, p_sc, albedo, f_corona, 1.0)

    assert terms.l_h_over_l_c == pytest.approx(expected_lh)
    assert terms.l_s_over_l_c == pytest.approx(expected_ls)
    assert terms.intrinsic_seed_over_l_c == pytest.approx(d_ratio)
