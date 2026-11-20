"""Tests for the allocation-light, moment-based and shared-path code paths.

Each optimised path is checked against a direct reference computation on the
same random draws, so the speedups cannot silently change the estimators.
"""

from __future__ import annotations

import numpy as np
import pytest

from mc_pricing import (
    OptionSpec,
    OptionType,
    black_scholes as bs,
    monte_carlo_greeks,
    monte_carlo_price,
    monte_carlo_price_to_accuracy,
)
from mc_pricing.engine import draw_normals, simulate_terminal
from mc_pricing.greeks import _discounted_payoff_per_path, fd_gamma, fd_rho
from mc_pricing.stats import CentredMoments, summarize, summarize_moments
from mc_pricing.variance_reduction import (
    apply_control_variate,
    collapse_antithetic,
    control_variate_moments,
    expected_terminal,
)


def _reference_price(spec, n_paths, antithetic, control_variate, seed):
    """The straightforward array-based pricer the fast path must reproduce."""
    rng = np.random.default_rng(seed)
    n = n_paths + (n_paths % 2) if antithetic else n_paths
    if antithetic:
        base = rng.standard_normal(n // 2)
        z = np.concatenate([base, -base])
    else:
        z = rng.standard_normal(n)
    s = spec
    terminal = s.spot * np.exp(
        (s.rate - s.dividend_yield - 0.5 * s.volatility**2) * s.maturity
        + s.volatility * np.sqrt(s.maturity) * z
    )
    w = s.option_type.sign
    y = np.exp(-s.rate * s.maturity) * np.maximum(w * (terminal - s.strike), 0.0)
    plain = summarize(y)
    x = terminal
    if antithetic:
        y, x = collapse_antithetic(y), collapse_antithetic(x)
    beta = 0.0
    if control_variate:
        beta = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)
        y = y - beta * (x - expected_terminal(s))
    res = summarize(y)
    return res, plain.std_error**2 / res.std_error**2, beta


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("antithetic", [False, True])
@pytest.mark.parametrize("control_variate", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 7_777])
def test_pricer_matches_reference(option_type, antithetic, control_variate, chunk_size):
    spec = OptionSpec(100, 95, 0.75, 0.04, 0.3, option_type, 0.01)
    ref, ref_ratio, ref_beta = _reference_price(
        spec, 100_001, antithetic, control_variate, seed=8
    )
    res = monte_carlo_price(
        spec,
        100_001,
        antithetic=antithetic,
        control_variate=control_variate,
        seed=8,
        chunk_size=chunk_size,
    )
    assert res.price == pytest.approx(ref.estimate, rel=1e-12)
    assert res.std_error == pytest.approx(ref.std_error, rel=1e-10)
    assert res.result.n_samples == ref.n_samples
    assert res.variance_reduction_ratio == pytest.approx(ref_ratio, rel=1e-10)
    assert res.control_beta == pytest.approx(ref_beta, rel=1e-10, abs=1e-15)


def test_plain_price_bit_identical_to_reference():
    spec = OptionSpec(100, 100, 1.0, 0.05, 0.2, OptionType.CALL)
    ref, _, _ = _reference_price(spec, 50_000, False, False, seed=3)
    assert monte_carlo_price(spec, 50_000, seed=3).price == ref.estimate


def test_invalid_chunk_and_paths():
    spec = OptionSpec(100, 100, 1.0, 0.05, 0.2, OptionType.CALL)
    with pytest.raises(ValueError):
        monte_carlo_price(spec, 1_000, chunk_size=0)
    with pytest.raises(ValueError):
        monte_carlo_price(spec, 0)


@pytest.mark.parametrize("n", [1, 2, 7, 10])
def test_draw_normals_antithetic_layout(n):
    z = draw_normals(n, np.random.default_rng(4), antithetic=True)
    half = (n + 1) // 2
    base = np.random.default_rng(4).standard_normal(half)
    assert np.array_equal(z, np.concatenate([base, -base])[:n])


def test_simulate_terminal_keeps_normals_and_is_exact():
    spec = OptionSpec(80, 100, 2.0, 0.03, 0.4, OptionType.CALL, 0.02)
    z = np.random.default_rng(0).standard_normal(1_001)
    z_copy = z.copy()
    terminal = simulate_terminal(spec, 0, None, normals=z).terminal_prices  # type: ignore[arg-type]
    expected = spec.spot * np.exp(
        (spec.rate - spec.dividend_yield - 0.5 * spec.volatility**2) * spec.maturity
        + spec.volatility * np.sqrt(spec.maturity) * z
    )
    assert np.array_equal(terminal, expected)
    assert np.array_equal(z, z_copy)


def test_moments_merge_matches_whole_sample():
    rng = np.random.default_rng(1)
    x = rng.normal(100.0, 5.0, 10_001)
    y = 3.0 * x + rng.normal(0.0, 1.0, x.size)
    whole = CentredMoments.from_arrays(y, x)
    merged = CentredMoments.from_arrays(y[:123], x[:123])
    for lo in range(123, x.size, 2_500):
        merged = merged.merge(CentredMoments.from_arrays(y[lo : lo + 2_500], x[lo : lo + 2_500]))
    for field in ("n", "mean_y", "mean_x", "m2_yy", "m2_xx", "m2_xy"):
        assert getattr(merged, field) == pytest.approx(getattr(whole, field), rel=1e-12)
    res = summarize_moments(whole.n, whole.mean_y, whole.m2_yy)
    ref = summarize(y)
    assert res.estimate == pytest.approx(ref.estimate, rel=1e-14)
    assert res.std_error == pytest.approx(ref.std_error, rel=1e-12)


def test_control_variate_moments_match_adjusted_samples():
    rng = np.random.default_rng(2)
    x = rng.lognormal(4.6, 0.2, 5_000)
    y = np.maximum(x - 100.0, 0.0)
    cv = apply_control_variate(y, x, 101.0)
    mean, m2, beta = control_variate_moments(CentredMoments.from_arrays(y, x), 101.0)
    ref = summarize(cv.adjusted)
    assert beta == pytest.approx(np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1), rel=1e-12)
    assert cv.beta == pytest.approx(beta, rel=1e-12)
    assert mean == pytest.approx(ref.estimate, rel=1e-12)
    assert summarize_moments(x.size, mean, m2).std_error == pytest.approx(
        ref.std_error, rel=1e-9
    )


def test_degenerate_control_moments_fall_back():
    y = np.array([1.0, 2.0, 3.0])
    x = np.full(3, 5.0)
    assert control_variate_moments(CentredMoments.from_arrays(y, x), 5.0)[2] == 0.0
    assert apply_control_variate(y, x, 5.0).beta == 0.0


def test_fd_gamma_and_rho_rescaling_match_repricing(atm_call):
    """Rescaled base prices give the same Gamma/Rho as full re-simulation."""
    z = draw_normals(100_000, np.random.default_rng(13), antithetic=True)
    h = atm_call.spot * 1e-2
    p0 = _discounted_payoff_per_path(atm_call, z)
    p_up = _discounted_payoff_per_path(atm_call.with_spot(atm_call.spot + h), z)
    p_dn = _discounted_payoff_per_path(atm_call.with_spot(atm_call.spot - h), z)
    gamma_ref = np.mean((p_up - 2.0 * p0 + p_dn) / (h * h))
    assert fd_gamma(atm_call, 100_000, seed=13).value == pytest.approx(gamma_ref, rel=1e-9)

    hr = 1e-4
    r_up = _discounted_payoff_per_path(atm_call.with_rate(atm_call.rate + hr), z)
    r_dn = _discounted_payoff_per_path(atm_call.with_rate(atm_call.rate - hr), z)
    rho_ref = np.mean((r_up - r_dn) / (2.0 * hr))
    assert fd_rho(atm_call, 100_000, seed=13).value == pytest.approx(rho_ref, rel=1e-9)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_shared_path_greeks_match_bs(option_type):
    spec = OptionSpec(100, 100, 1.0, 0.05, 0.2, option_type, 0.03)
    mc = monte_carlo_greeks(spec, 400_000, seed=21, shared_paths=True)
    cf = bs.greeks(spec)
    assert abs(mc["delta"].value - cf["delta"]) < 4 * mc["delta"].std_error + 1e-3
    assert abs(mc["vega"].value - cf["vega"]) < 4 * mc["vega"].std_error + 1e-2
    assert mc["gamma"].value == pytest.approx(cf["gamma"], abs=3e-3)
    assert abs(mc["theta"].value - cf["theta"]) < 4 * mc["theta"].std_error + 5e-2
    assert abs(mc["rho"].value - cf["rho"]) < 4 * mc["rho"].std_error + 5e-2
    assert mc["delta"].method == "pathwise"
    assert mc["gamma"].method == "finite-difference-crn"


def test_shared_path_greeks_near_expiry():
    spec = OptionSpec(100, 100, 0.02, 0.05, 0.2, OptionType.CALL)
    mc = monte_carlo_greeks(spec, 400_000, seed=22, shared_paths=True)
    cf = bs.greeks(spec)
    assert abs(mc["delta"].value - cf["delta"]) < 4 * mc["delta"].std_error + 1e-3
    assert abs(mc["theta"].value - cf["theta"]) < 4 * mc["theta"].std_error + 5e-1


def test_price_to_accuracy_hits_target(atm_call):
    target = 5e-3
    res = monte_carlo_price_to_accuracy(atm_call, target, seed=30)
    assert res.antithetic and res.control_variate
    assert res.std_error < 1.15 * target
    assert abs(res.price - bs.price(atm_call)) < 4 * res.std_error


def test_price_to_accuracy_plain_needs_more_paths(atm_call):
    fast = monte_carlo_price_to_accuracy(atm_call, 2e-2, seed=31)
    plain = monte_carlo_price_to_accuracy(
        atm_call, 2e-2, antithetic=False, control_variate=False, seed=31
    )
    assert plain.n_paths_simulated > 5 * fast.n_paths_simulated
    assert plain.std_error < 1.15 * 2e-2


def test_price_to_accuracy_caps_and_validates(atm_call):
    with pytest.warns(UserWarning, match="max_paths"):
        res = monte_carlo_price_to_accuracy(atm_call, 1e-6, max_paths=10_000, seed=32)
    assert res.n_paths_simulated == 10_000
    with pytest.raises(ValueError):
        monte_carlo_price_to_accuracy(atm_call, 0.0)
