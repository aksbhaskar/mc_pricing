"""Tests that Monte Carlo Greek estimators agree with the closed-form Greeks."""

from __future__ import annotations

import pytest

from mc_pricing import OptionSpec, OptionType, black_scholes as bs
from mc_pricing.greeks import (
    fd_gamma,
    fd_rho,
    fd_theta,
    monte_carlo_greeks,
    pathwise_delta,
    pathwise_vega,
)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_pathwise_delta_matches_bs(market_params, option_type):
    S, K, T, r, sigma, q = market_params
    spec = OptionSpec(S, K, T, r, sigma, option_type, q)
    est = pathwise_delta(spec, 400_000, seed=11)
    assert abs(est.value - bs.delta(spec)) < 4 * est.std_error + 1e-3


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_pathwise_vega_matches_bs(market_params, option_type):
    S, K, T, r, sigma, q = market_params
    spec = OptionSpec(S, K, T, r, sigma, option_type, q)
    est = pathwise_vega(spec, 400_000, seed=12)
    assert abs(est.value - bs.vega(spec)) < 4 * est.std_error + 1e-2


def test_fd_gamma_matches_bs(atm_call):
    est = fd_gamma(atm_call, 500_000, seed=13)
    # Gamma finite differences are noisy; allow a modest absolute band.
    assert est.value == pytest.approx(bs.gamma(atm_call), abs=3e-3)


def test_fd_theta_matches_bs(atm_call):
    est = fd_theta(atm_call, 500_000, seed=14)
    assert abs(est.value - bs.theta(atm_call)) < 4 * est.std_error + 5e-2


def test_fd_rho_matches_bs(atm_call):
    est = fd_rho(atm_call, 500_000, seed=15)
    assert abs(est.value - bs.rho(atm_call)) < 4 * est.std_error + 5e-2


def test_monte_carlo_greeks_bundle(atm_put):
    """The convenience bundle returns all five Greeks, each close to BS."""
    mc = monte_carlo_greeks(atm_put, 400_000, seed=16)
    cf = bs.greeks(atm_put)
    assert set(mc) == {"delta", "gamma", "vega", "theta", "rho"}

    assert abs(mc["delta"].value - cf["delta"]) < 4 * mc["delta"].std_error + 1e-3
    assert abs(mc["vega"].value - cf["vega"]) < 4 * mc["vega"].std_error + 1e-2
    assert mc["gamma"].value == pytest.approx(cf["gamma"], abs=3e-3)
    assert abs(mc["theta"].value - cf["theta"]) < 4 * mc["theta"].std_error + 5e-2
    assert abs(mc["rho"].value - cf["rho"]) < 4 * mc["rho"].std_error + 5e-2


def test_greek_methods_labelled(atm_call):
    mc = monte_carlo_greeks(atm_call, 50_000, seed=1)
    assert mc["delta"].method == "pathwise"
    assert mc["vega"].method == "pathwise"
    assert mc["gamma"].method == "finite-difference-crn"
    assert mc["theta"].method == "finite-difference-crn"
    assert mc["rho"].method == "finite-difference-crn"
