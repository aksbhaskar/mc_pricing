"""Tests for the OptionSpec value object: validation and functional updates."""

from __future__ import annotations

import pytest

from mc_pricing import OptionSpec, OptionType


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spot": 0.0},
        {"spot": -1.0},
        {"strike": 0.0},
        {"strike": -5.0},
        {"maturity": 0.0},
        {"maturity": -0.5},
        {"volatility": -0.1},
    ],
)
def test_invalid_parameters_raise(kwargs):
    base = dict(spot=100, strike=100, maturity=1.0, rate=0.05, volatility=0.2)
    base.update(kwargs)
    with pytest.raises(ValueError):
        OptionSpec(**base)


def test_string_option_type_is_coerced():
    spec = OptionSpec(100, 100, 1.0, 0.05, 0.2, option_type="put")
    assert spec.option_type is OptionType.PUT
    assert not spec.is_call


def test_option_type_sign():
    assert OptionType.CALL.sign == 1
    assert OptionType.PUT.sign == -1


def test_functional_updates_return_new_spec():
    spec = OptionSpec(100, 100, 1.0, 0.05, 0.2, OptionType.CALL, 0.01)
    bumped = spec.with_spot(101.0)
    assert bumped.spot == 101.0
    assert spec.spot == 100.0  # original untouched (frozen)
    # everything else preserved
    assert bumped.dividend_yield == spec.dividend_yield
    assert bumped.option_type is spec.option_type


def test_spec_is_frozen():
    spec = OptionSpec(100, 100, 1.0, 0.05, 0.2)
    with pytest.raises(Exception):
        spec.spot = 200.0  # type: ignore[misc]
