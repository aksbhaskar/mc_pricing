"""Tests for the closed-form Black-Scholes-Merton layer.

These cover known textbook values, put-call parity, the analytical Greeks
against high-accuracy finite differences, and the degenerate limits.
"""

from __future__ import annotations

import math

import pytest

from mc_pricing import OptionSpec, OptionType, black_scholes as bs


def test_known_atm_call_value():
    """Textbook value: S=K=100, r=5%, sigma=20%, T=1, q=0 -> ~10.4506."""
    spec = OptionSpec(100, 100, 1.0, 0.05, 0.20, OptionType.CALL)
    assert bs.price(spec) == pytest.approx(10.450583, abs=1e-5)


def test_known_atm_put_value():
    """Same parameters, put leg -> ~5.5735 (via parity)."""
    spec = OptionSpec(100, 100, 1.0, 0.05, 0.20, OptionType.PUT)
    assert bs.price(spec) == pytest.approx(5.573526, abs=1e-5)


def test_put_call_parity(market_params):
    """C - P must equal S e^{-qT} - K e^{-rT} across regimes."""
    S, K, T, r, sigma, q = market_params
    call = OptionSpec(S, K, T, r, sigma, OptionType.CALL, q)
    put = OptionSpec(S, K, T, r, sigma, OptionType.PUT, q)
    lhs = bs.price(call) - bs.price(put)
    rhs = bs.put_call_parity_rhs(call)
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_price_is_non_negative(market_params):
    S, K, T, r, sigma, q = market_params
    for ot in (OptionType.CALL, OptionType.PUT):
        assert bs.price(OptionSpec(S, K, T, r, sigma, ot, q)) >= 0.0


@pytest.mark.parametrize("greek", ["delta", "gamma", "vega", "theta", "rho"])
def test_greeks_match_finite_difference(market_params, greek):
    """Each analytical Greek matches a central finite difference of the price."""
    S, K, T, r, sigma, q = market_params
    spec = OptionSpec(S, K, T, r, sigma, OptionType.CALL, q)

    def price_with(**changes) -> float:
        return bs.price(spec._replace(**changes))

    if greek == "delta":
        h = 1e-4 * S
        fd = (price_with(spot=S + h) - price_with(spot=S - h)) / (2 * h)
    elif greek == "gamma":
        h = 1e-3 * S
        fd = (price_with(spot=S + h) - 2 * bs.price(spec) + price_with(spot=S - h)) / h**2
    elif greek == "vega":
        h = 1e-5
        fd = (price_with(volatility=sigma + h) - price_with(volatility=sigma - h)) / (2 * h)
    elif greek == "theta":
        h = 1e-5
        fd = -(price_with(maturity=T + h) - price_with(maturity=T - h)) / (2 * h)
    else:  # rho
        h = 1e-6
        fd = (price_with(rate=r + h) - price_with(rate=r - h)) / (2 * h)

    analytic = getattr(bs, greek)(spec)
    assert analytic == pytest.approx(fd, rel=1e-4, abs=1e-4)


def test_call_delta_bounds(market_params):
    """Call delta in (0, e^{-qT}); put delta in (-e^{-qT}, 0)."""
    S, K, T, r, sigma, q = market_params
    disc_q = math.exp(-q * T)
    call = OptionSpec(S, K, T, r, sigma, OptionType.CALL, q)
    put = OptionSpec(S, K, T, r, sigma, OptionType.PUT, q)
    assert 0.0 < bs.delta(call) < disc_q + 1e-12
    assert -disc_q - 1e-12 < bs.delta(put) < 0.0


def test_gamma_and_vega_equal_for_call_and_put(market_params):
    """Gamma and Vega are identical for a call and put with the same params."""
    S, K, T, r, sigma, q = market_params
    call = OptionSpec(S, K, T, r, sigma, OptionType.CALL, q)
    put = OptionSpec(S, K, T, r, sigma, OptionType.PUT, q)
    assert bs.gamma(call) == pytest.approx(bs.gamma(put), abs=1e-12)
    assert bs.vega(call) == pytest.approx(bs.vega(put), abs=1e-12)


def test_zero_volatility_limit_call():
    """With sigma -> 0 the call is worth its discounted intrinsic on the forward."""
    S, K, T, r, q = 100.0, 90.0, 1.0, 0.05, 0.0
    spec = OptionSpec(S, K, T, r, 1e-16, OptionType.CALL, q)
    forward = S * math.exp((r - q) * T)
    expected = math.exp(-r * T) * max(forward - K, 0.0)
    assert bs.price(spec) == pytest.approx(expected, abs=1e-8)
    # Gamma and Vega collapse to zero in the deterministic limit.
    assert bs.gamma(spec) == pytest.approx(0.0, abs=1e-10)
    assert bs.vega(spec) == pytest.approx(0.0, abs=1e-10)


def test_deep_itm_call_delta_approaches_disc_q():
    """A deeply in-the-money call has delta close to e^{-qT}."""
    spec = OptionSpec(200, 100, 1.0, 0.05, 0.20, OptionType.CALL, 0.02)
    assert bs.delta(spec) == pytest.approx(math.exp(-0.02), abs=1e-3)
