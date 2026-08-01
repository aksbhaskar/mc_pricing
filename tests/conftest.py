"""Shared pytest fixtures for the mc_pricing test suite."""

from __future__ import annotations

import pytest

from mc_pricing import OptionSpec, OptionType


@pytest.fixture
def atm_call() -> OptionSpec:
    """A canonical at-the-money 1-year call with a dividend yield."""
    return OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        option_type=OptionType.CALL,
        dividend_yield=0.03,
    )


@pytest.fixture
def atm_put() -> OptionSpec:
    """The put with identical parameters to :func:`atm_call`."""
    return OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        option_type=OptionType.PUT,
        dividend_yield=0.03,
    )


@pytest.fixture(
    params=[
        # (spot, strike, T, r, sigma, q)
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0),
        (100.0, 90.0, 0.5, 0.03, 0.35, 0.02),   # in-the-money call
        (100.0, 120.0, 2.0, 0.01, 0.15, 0.04),  # out-of-the-money call
        (50.0, 50.0, 0.25, 0.08, 0.50, 0.0),    # short-dated, high vol
    ]
)
def market_params(request) -> tuple[float, float, float, float, float, float]:
    """A spread of market regimes: ATM, ITM, OTM, short-dated/high-vol."""
    return request.param
