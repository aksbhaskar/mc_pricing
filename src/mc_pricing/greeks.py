r"""Monte Carlo Greek estimators.

Two families of estimator are provided, chosen per-Greek according to what
works best for a vanilla European payoff:

Pathwise derivatives (Delta, Vega)
----------------------------------
When the discounted payoff is (almost everywhere) differentiable in the
parameter, we may differentiate *inside* the expectation and estimate the
derivative directly on each path. Interchanging :math:`\partial` and
:math:`\mathbb{E}` is justified because the call/put payoff is Lipschitz and
differentiable except on the measure-zero set :math:`\{S_T = K\}`.

For Delta, with :math:`\partial S_T/\partial S_0 = S_T/S_0`,

.. math::

    \Delta = e^{-rT}\,\mathbb{E}\!\left[\omega\,\frac{S_T}{S_0}\,
        \mathbf{1}\{\omega(S_T - K) > 0\}\right].

For Vega, with :math:`\partial S_T/\partial\sigma = S_T(\sqrt T\,Z - \sigma T)`,

.. math::

    \mathcal{V} = e^{-rT}\,\mathbb{E}\!\left[\omega\,S_T(\sqrt T\,Z - \sigma T)\,
        \mathbf{1}\{\omega(S_T - K) > 0\}\right].

Pathwise estimators are unbiased and typically far lower variance than finite
differences, and they need no bump size.

Finite differences with common random numbers (Gamma, Theta, Rho)
-----------------------------------------------------------------
Gamma is a *second* derivative (the pathwise first derivative, an indicator,
is no longer differentiable), and Theta/Rho are conventionally quoted via
re-pricing, so we bump-and-reprice. Crucially the bumped and base prices are
computed from the **same** normals :math:`Z` (common random numbers): the
random component cancels in the difference, leaving an estimator whose variance
is :math:`O(1)` rather than :math:`O(1/h^2)`. Central differences are used for
:math:`O(h^2)` bias:

.. math::

    \Gamma \approx \frac{V(S_0+h) - 2V(S_0) + V(S_0-h)}{h^2},\qquad
    \Theta \approx -\frac{V(T+h_T) - V(T-h_T)}{2h_T},\qquad
    \rho \approx \frac{V(r+h_r) - V(r-h_r)}{2h_r}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .engine import draw_normals, simulate_terminal
from .option import OptionSpec
from .payoff import terminal_payoff
from .stats import MCResult, summarize


@dataclass(frozen=True, slots=True)
class GreekEstimate:
    """A single Monte Carlo Greek with its uncertainty and method label.

    Attributes
    ----------
    value:
        The point estimate of the Greek.
    std_error:
        Standard error of the estimate.
    method:
        Either ``"pathwise"`` or ``"finite-difference-crn"``.
    """

    value: float
    std_error: float
    method: str

    @classmethod
    def from_result(cls, result: MCResult, method: str) -> "GreekEstimate":
        return cls(value=result.estimate, std_error=result.std_error, method=method)


def _discounted_payoff(
    spec: OptionSpec, terminal_prices: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Discounted payoff of ``spec`` on each of the given terminal prices."""
    payoff = terminal_payoff(spec, terminal_prices)
    payoff *= np.exp(-spec.rate * spec.maturity)
    return payoff


def _terminal(spec: OptionSpec, z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Terminal prices of ``spec`` driven by the shared normals ``z``."""
    return simulate_terminal(spec, z.size, rng=None, normals=z).terminal_prices  # type: ignore[arg-type]


def _discounted_payoff_per_path(
    spec: OptionSpec, z: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Discounted payoff on each path for a given spec and shared normals ``z``.

    This is the workhorse for the common-random-number finite differences:
    every bumped valuation calls it with the *same* ``z`` so the stochastic
    part cancels in the differences.
    """
    return _discounted_payoff(spec, _terminal(spec, z))


def _delta_per_path(spec: OptionSpec, S_T: NDArray[np.float64]) -> NDArray[np.float64]:
    w = spec.option_type.sign
    in_money = (w * (S_T - spec.strike)) > 0.0
    discount = np.exp(-spec.rate * spec.maturity)
    return discount * w * (S_T / spec.spot) * in_money


def _vega_per_path(
    spec: OptionSpec, S_T: NDArray[np.float64], z: NDArray[np.float64]
) -> NDArray[np.float64]:
    w = spec.option_type.sign
    in_money = (w * (S_T - spec.strike)) > 0.0
    sqrt_t = np.sqrt(spec.maturity)
    dS_dsigma = S_T * (sqrt_t * z - spec.volatility * spec.maturity)
    discount = np.exp(-spec.rate * spec.maturity)
    return discount * w * dS_dsigma * in_money


def _gamma_per_path(
    spec: OptionSpec, S_T: NDArray[np.float64], rel_bump: float
) -> NDArray[np.float64]:
    # S_T is linear in S_0 for fixed Z, so the bumped terminal prices are the
    # base ones rescaled: no further exponentials needed.
    h = spec.spot * rel_bump
    p0 = _discounted_payoff(spec, S_T)
    p_up = _discounted_payoff(spec.with_spot(spec.spot + h), S_T * ((spec.spot + h) / spec.spot))
    p_dn = _discounted_payoff(spec.with_spot(spec.spot - h), S_T * ((spec.spot - h) / spec.spot))
    return (p_up - 2.0 * p0 + p_dn) / (h * h)


def _theta_per_path(
    spec: OptionSpec, z: NDArray[np.float64], rel_bump: float
) -> NDArray[np.float64]:
    h = spec.maturity * rel_bump
    p_up = _discounted_payoff_per_path(spec.with_maturity(spec.maturity + h), z)
    p_dn = _discounted_payoff_per_path(spec.with_maturity(spec.maturity - h), z)
    # theta = -dV/dT
    return -(p_up - p_dn) / (2.0 * h)


def _rho_per_path(
    spec: OptionSpec, S_T: NDArray[np.float64], abs_bump: float
) -> NDArray[np.float64]:
    # A rate bump of h scales every S_T by exp(±hT) for fixed Z; the discount
    # factor change is picked up by pricing under the bumped spec.
    h = abs_bump
    growth = np.exp(h * spec.maturity)
    p_up = _discounted_payoff(spec.with_rate(spec.rate + h), S_T * growth)
    p_dn = _discounted_payoff(spec.with_rate(spec.rate - h), S_T / growth)
    return (p_up - p_dn) / (2.0 * h)


def _crn_estimate(est: NDArray[np.float64], confidence: float) -> GreekEstimate:
    return GreekEstimate.from_result(summarize(est, confidence), "finite-difference-crn")


def _pathwise_estimate(est: NDArray[np.float64], confidence: float) -> GreekEstimate:
    return GreekEstimate.from_result(summarize(est, confidence), "pathwise")


def _normals(
    n_paths: int,
    antithetic: bool,
    seed: int | None,
    rng: np.random.Generator | None,
) -> NDArray[np.float64]:
    if rng is None:
        rng = np.random.default_rng(seed)
    if antithetic:
        n_paths += n_paths % 2
    return draw_normals(n_paths, rng, antithetic=antithetic)


def pathwise_delta(
    spec: OptionSpec,
    n_paths: int = 100_000,
    *,
    antithetic: bool = True,
    confidence: float = 0.95,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> GreekEstimate:
    r"""Estimate Delta by the pathwise method.

    Per-path estimator
    :math:`e^{-rT}\,\omega\,(S_T/S_0)\,\mathbf{1}\{\omega(S_T-K)>0\}`.
    """
    z = _normals(n_paths, antithetic, seed, rng)
    return _pathwise_estimate(_delta_per_path(spec, _terminal(spec, z)), confidence)


def pathwise_vega(
    spec: OptionSpec,
    n_paths: int = 100_000,
    *,
    antithetic: bool = True,
    confidence: float = 0.95,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> GreekEstimate:
    r"""Estimate Vega by the pathwise method.

    Per-path estimator
    :math:`e^{-rT}\,\omega\,S_T(\sqrt T\,Z - \sigma T)\,
    \mathbf{1}\{\omega(S_T-K)>0\}`, using the stored driving normals :math:`Z`.
    """
    z = _normals(n_paths, antithetic, seed, rng)
    return _pathwise_estimate(_vega_per_path(spec, _terminal(spec, z), z), confidence)


def fd_gamma(
    spec: OptionSpec,
    n_paths: int = 200_000,
    *,
    rel_bump: float = 1e-2,
    antithetic: bool = True,
    confidence: float = 0.95,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> GreekEstimate:
    r"""Estimate Gamma by a central finite difference in spot with CRN.

    Per-path estimator :math:`(P_+ - 2P_0 + P_-)/h^2` where the three prices
    share the same normals. ``rel_bump`` sets :math:`h = \text{rel\_bump}\times
    S_0`. Because :math:`S_T` is proportional to :math:`S_0` for fixed
    :math:`Z`, the bumped terminal prices are rescaled base prices.
    """
    z = _normals(n_paths, antithetic, seed, rng)
    return _crn_estimate(_gamma_per_path(spec, _terminal(spec, z), rel_bump), confidence)


def fd_theta(
    spec: OptionSpec,
    n_paths: int = 200_000,
    *,
    rel_bump: float = 1e-3,
    antithetic: bool = True,
    confidence: float = 0.95,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> GreekEstimate:
    r"""Estimate Theta by a central finite difference in maturity with CRN.

    Theta is calendar-time decay :math:`-\partial V/\partial T`, so the sign of
    the maturity difference is negated. ``rel_bump`` sets
    :math:`h_T = \text{rel\_bump}\times T`.
    """
    z = _normals(n_paths, antithetic, seed, rng)
    return _crn_estimate(_theta_per_path(spec, z, rel_bump), confidence)


def fd_rho(
    spec: OptionSpec,
    n_paths: int = 200_000,
    *,
    abs_bump: float = 1e-4,
    antithetic: bool = True,
    confidence: float = 0.95,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> GreekEstimate:
    r"""Estimate Rho by a central finite difference in the rate with CRN.

    Per-path estimator :math:`(P_{r+h} - P_{r-h})/(2h)`; bumping :math:`r`
    shifts both the risk-neutral drift (a factor :math:`e^{\pm hT}` on each
    :math:`S_T`) and the discount factor, both applied on the shared normals.
    """
    z = _normals(n_paths, antithetic, seed, rng)
    return _crn_estimate(_rho_per_path(spec, _terminal(spec, z), abs_bump), confidence)


def monte_carlo_greeks(
    spec: OptionSpec,
    n_paths: int = 200_000,
    *,
    antithetic: bool = True,
    confidence: float = 0.95,
    seed: int | None = None,
    shared_paths: bool = False,
) -> dict[str, GreekEstimate]:
    """Estimate all five Greeks by Monte Carlo in one call.

    Delta and Vega use pathwise estimators; Gamma, Theta and Rho use
    common-random-number central finite differences. A shared ``seed`` is
    threaded through so results are reproducible.

    Parameters
    ----------
    shared_paths:
        ``False`` (default): each Greek draws its own independent normals, so
        the five estimates are statistically independent. ``True``: all five
        are computed from one set of normals and one set of base terminal
        prices (spot and rate bumps are rescalings of them; only Theta's
        maturity bumps need new exponentials). This does roughly a fifth of
        the random draws and a third of the exponentials, so it is markedly
        faster, but the estimates become correlated with each other and the
        seeded values differ from the default mode. Each Greek on its own is
        exactly as accurate as in the default mode.

    Returns
    -------
    dict[str, GreekEstimate]
        Keys ``delta``, ``gamma``, ``vega``, ``theta``, ``rho``.
    """
    if shared_paths:
        z = _normals(n_paths, antithetic, seed, None)
        S_T = _terminal(spec, z)
        return {
            "delta": _pathwise_estimate(_delta_per_path(spec, S_T), confidence),
            "gamma": _crn_estimate(_gamma_per_path(spec, S_T, 1e-2), confidence),
            "vega": _pathwise_estimate(_vega_per_path(spec, S_T, z), confidence),
            "theta": _crn_estimate(_theta_per_path(spec, z, 1e-3), confidence),
            "rho": _crn_estimate(_rho_per_path(spec, S_T, 1e-4), confidence),
        }

    ss = np.random.SeedSequence(seed)
    streams = [np.random.default_rng(s) for s in ss.spawn(5)]
    return {
        "delta": pathwise_delta(
            spec, n_paths, antithetic=antithetic, confidence=confidence, rng=streams[0]
        ),
        "gamma": fd_gamma(
            spec, n_paths, antithetic=antithetic, confidence=confidence, rng=streams[1]
        ),
        "vega": pathwise_vega(
            spec, n_paths, antithetic=antithetic, confidence=confidence, rng=streams[2]
        ),
        "theta": fd_theta(
            spec, n_paths, antithetic=antithetic, confidence=confidence, rng=streams[3]
        ),
        "rho": fd_rho(
            spec, n_paths, antithetic=antithetic, confidence=confidence, rng=streams[4]
        ),
    }
