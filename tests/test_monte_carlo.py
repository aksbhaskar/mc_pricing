"""Tests for the Monte Carlo pricer.

Focus: statistical correctness (MC converges to Black-Scholes within its
confidence interval), error decay with path count, reproducibility, and
put-call parity of the estimator.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mc_pricing import OptionSpec, OptionType, black_scholes as bs, monte_carlo_price


@pytest.mark.parametrize("antithetic", [False, True])
@pytest.mark.parametrize("control_variate", [False, True])
def test_mc_within_confidence_interval(market_params, antithetic, control_variate):
    """The analytic price falls inside the MC 99% CI for every technique."""
    S, K, T, r, sigma, q = market_params
    spec = OptionSpec(S, K, T, r, sigma, OptionType.CALL, q)
    analytic = bs.price(spec)
    res = monte_carlo_price(
        spec,
        n_paths=400_000,
        antithetic=antithetic,
        control_variate=control_variate,
        confidence=0.99,
        seed=12345,
    )
    lo, hi = res.confidence_interval
    assert lo <= analytic <= hi


def test_mc_error_decays_as_sqrt_n():
    """Standard error should roughly halve when paths quadruple (~1/sqrt(N))."""
    spec = OptionSpec(100, 100, 1.0, 0.05, 0.20, OptionType.CALL)
    se_small = monte_carlo_price(spec, n_paths=50_000, seed=7).std_error
    se_large = monte_carlo_price(spec, n_paths=200_000, seed=7).std_error
    ratio = se_small / se_large
    # Expect ~2.0; allow a generous statistical band.
    assert 1.6 < ratio < 2.5


def test_reproducible_with_seed():
    """Same seed -> identical estimate; different seed -> (almost surely) different."""
    spec = OptionSpec(100, 100, 1.0, 0.05, 0.20, OptionType.CALL)
    a = monte_carlo_price(spec, n_paths=50_000, seed=42)
    b = monte_carlo_price(spec, n_paths=50_000, seed=42)
    c = monte_carlo_price(spec, n_paths=50_000, seed=43)
    assert a.price == b.price
    assert a.price != c.price


def test_explicit_rng_matches_seed():
    """Passing an explicit Generator is equivalent to passing its seed."""
    spec = OptionSpec(100, 100, 1.0, 0.05, 0.20, OptionType.CALL)
    from_seed = monte_carlo_price(spec, n_paths=20_000, seed=99)
    from_rng = monte_carlo_price(
        spec, n_paths=20_000, rng=np.random.default_rng(99)
    )
    assert from_seed.price == from_rng.price


def test_mc_put_call_parity():
    """Estimated C - P should match the parity RHS within combined error."""
    call = OptionSpec(100, 105, 1.0, 0.05, 0.25, OptionType.CALL, 0.01)
    put = OptionSpec(100, 105, 1.0, 0.05, 0.25, OptionType.PUT, 0.01)
    rng_seed = 2024
    c = monte_carlo_price(call, 400_000, antithetic=True, seed=rng_seed)
    p = monte_carlo_price(put, 400_000, antithetic=True, seed=rng_seed)
    parity = bs.put_call_parity_rhs(call)
    combined_se = math.hypot(c.std_error, p.std_error)
    assert abs((c.price - p.price) - parity) < 4 * combined_se


def test_odd_n_paths_rounds_up_for_antithetic():
    """An odd path count is rounded up to keep antithetic pairs balanced."""
    spec = OptionSpec(100, 100, 1.0, 0.05, 0.20, OptionType.CALL)
    res = monte_carlo_price(spec, n_paths=9999, antithetic=True, seed=1)
    assert res.n_paths_simulated == 10000


def test_plain_ratio_is_one():
    """A plain run reports a variance-reduction ratio of exactly 1."""
    spec = OptionSpec(100, 100, 1.0, 0.05, 0.20, OptionType.CALL)
    res = monte_carlo_price(spec, n_paths=50_000, seed=1)
    assert res.variance_reduction_ratio == pytest.approx(1.0)
