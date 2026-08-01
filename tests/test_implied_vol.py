"""Tests for the Brent's-method implied-volatility solver."""

from __future__ import annotations

import pytest

from mc_pricing import OptionSpec, OptionType, black_scholes as bs, implied_volatility


@pytest.mark.parametrize("sigma", [0.05, 0.15, 0.30, 0.60, 1.20])
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_implied_vol_round_trip(sigma, option_type):
    """Pricing at sigma then inverting recovers sigma to solver tolerance."""
    spec = OptionSpec(100, 105, 0.75, 0.04, sigma, option_type, 0.02)
    price = bs.price(spec)
    iv = implied_volatility(
        price, 100, 105, 0.75, 0.04, option_type, dividend_yield=0.02
    )
    assert iv == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_accepts_string_type():
    spec = OptionSpec(100, 100, 1.0, 0.05, 0.22, OptionType.CALL)
    price = bs.price(spec)
    iv = implied_volatility(price, 100, 100, 1.0, 0.05, "call")
    assert iv == pytest.approx(0.22, abs=1e-6)


def test_implied_vol_raises_below_intrinsic():
    """A price below the no-arbitrage floor has no implied vol root."""
    with pytest.raises(ValueError):
        # Deep ITM call cannot be worth less than ~ its discounted intrinsic.
        implied_volatility(0.01, 200, 100, 1.0, 0.05, OptionType.CALL)
