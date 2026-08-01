"""Tests that variance-reduction techniques actually reduce variance and stay
unbiased.
"""

from __future__ import annotations

import numpy as np
import pytest

from mc_pricing import OptionSpec, OptionType, black_scholes as bs, monte_carlo_price
from mc_pricing.variance_reduction import (
    apply_control_variate,
    collapse_antithetic,
    expected_terminal,
)


def test_antithetic_reduces_variance(market_params):
    """Antithetic sampling yields SE no larger than plain MC (usually smaller)."""
    S, K, T, r, sigma, q = market_params
    spec = OptionSpec(S, K, T, r, sigma, OptionType.CALL, q)
    plain = monte_carlo_price(spec, 200_000, seed=5)
    anti = monte_carlo_price(spec, 200_000, antithetic=True, seed=5)
    assert anti.std_error < plain.std_error
    assert anti.variance_reduction_ratio > 1.0


def test_control_variate_reduces_variance(market_params):
    """The terminal-price control variate reduces variance."""
    S, K, T, r, sigma, q = market_params
    spec = OptionSpec(S, K, T, r, sigma, OptionType.CALL, q)
    plain = monte_carlo_price(spec, 200_000, seed=6)
    cv = monte_carlo_price(spec, 200_000, control_variate=True, seed=6)
    assert cv.std_error < plain.std_error
    assert cv.variance_reduction_ratio > 1.0


def test_combined_is_best(atm_call):
    """Antithetic + control variate beats either technique alone."""
    plain = monte_carlo_price(atm_call, 200_000, seed=8)
    anti = monte_carlo_price(atm_call, 200_000, antithetic=True, seed=8)
    cv = monte_carlo_price(atm_call, 200_000, control_variate=True, seed=8)
    both = monte_carlo_price(
        atm_call, 200_000, antithetic=True, control_variate=True, seed=8
    )
    assert both.std_error < anti.std_error
    assert both.std_error < cv.std_error
    assert both.variance_reduction_ratio > plain.variance_reduction_ratio


def test_variance_reduction_stays_unbiased(atm_call):
    """Every technique's estimate agrees with the analytic price within CI."""
    analytic = bs.price(atm_call)
    for kwargs in (
        {},
        {"antithetic": True},
        {"control_variate": True},
        {"antithetic": True, "control_variate": True},
    ):
        res = monte_carlo_price(atm_call, 400_000, confidence=0.99, seed=3, **kwargs)
        lo, hi = res.confidence_interval
        assert lo <= analytic <= hi, kwargs


def test_collapse_antithetic_preserves_mean():
    """Averaging antithetic pairs preserves the sample mean."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(1000)
    values = np.concatenate([base, -base])
    collapsed = collapse_antithetic(values)
    assert collapsed.size == 1000
    assert collapsed.mean() == pytest.approx(values.mean(), abs=1e-12)


def test_collapse_antithetic_rejects_odd_length():
    with pytest.raises(ValueError):
        collapse_antithetic(np.arange(5.0))


def test_control_variate_beta_zero_for_constant_control(atm_call):
    """A degenerate (zero-variance) control leaves the payoffs untouched."""
    payoffs = np.array([1.0, 2.0, 3.0, 4.0])
    controls = np.full(4, 7.0)  # constant -> zero variance
    res = apply_control_variate(payoffs, controls, control_mean=7.0)
    assert res.beta == 0.0
    np.testing.assert_allclose(res.adjusted, payoffs)


def test_expected_terminal_formula():
    """E[S_T] = S0 exp((r - q) T)."""
    spec = OptionSpec(100, 100, 2.0, 0.05, 0.20, OptionType.CALL, 0.03)
    assert expected_terminal(spec) == pytest.approx(100 * np.exp((0.05 - 0.03) * 2.0))
