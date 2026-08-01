r"""Closed-form Black-Scholes-Merton pricing and Greeks.

This module is the analytical ground truth against which every Monte Carlo
estimate in the package is validated.

The model
---------
Under the risk-neutral measure :math:`\mathbb{Q}` the underlying follows the
geometric Brownian motion

.. math::

    dS_t = (r - q) S_t\,dt + \sigma S_t\,dW_t,

so the terminal price is log-normal,

.. math::

    S_T = S_0 \exp\!\Big[(r - q - \tfrac12\sigma^2)T + \sigma\sqrt{T}\,Z\Big],
    \qquad Z \sim \mathcal{N}(0, 1).

Discounting the risk-neutral expectation of the payoff yields the
Black-Scholes-Merton formulae with continuous dividend yield :math:`q`:

.. math::

    d_1 = \frac{\ln(S_0/K) + (r - q + \tfrac12\sigma^2)T}{\sigma\sqrt T},
    \qquad d_2 = d_1 - \sigma\sqrt T,

    C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2),
    \qquad
    P = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .option import OptionSpec, OptionType

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    r"""Standard-normal density :math:`\phi(x) = e^{-x^2/2}/\sqrt{2\pi}`."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _norm_cdf(x: float) -> float:
    r"""Standard-normal CDF :math:`N(x)` via the error function.

    Uses :math:`N(x) = \tfrac12\,\mathrm{erfc}(-x/\sqrt2)`, which is the
    numerically stable form (``erfc`` avoids catastrophic cancellation in the
    far tails).
    """
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@dataclass(frozen=True, slots=True)
class _D:
    """Container for the intermediate :math:`d_1, d_2` and useful factors."""

    d1: float
    d2: float
    disc_r: float  # e^{-rT}
    disc_q: float  # e^{-qT}
    sqrt_t: float  # sqrt(T)


def _compute_d(spec: OptionSpec) -> _D:
    r"""Compute :math:`d_1, d_2` and the discount factors for a spec.

    Handles the degenerate zero-volatility limit (:math:`\sigma\sqrt T \to 0`)
    by pushing :math:`d_1, d_2` to :math:`\pm\infty` according to the sign of
    the log-moneyness of the *forward*, so the formulae collapse to the
    correct discounted-intrinsic value.
    """
    S, K, T = spec.spot, spec.strike, spec.maturity
    r, q, sigma = spec.rate, spec.dividend_yield, spec.volatility
    sqrt_t = math.sqrt(T)
    vol_sqrt_t = sigma * sqrt_t

    if vol_sqrt_t < 1e-15:
        # Deterministic forward F = S e^{(r-q)T}; option is worth its
        # discounted intrinsic value. Encode that via infinite d's.
        forward = S * math.exp((r - q) * T)
        if forward > K:
            d1 = d2 = math.inf
        elif forward < K:
            d1 = d2 = -math.inf
        else:
            d1 = d2 = 0.0
    else:
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t

    return _D(
        d1=d1,
        d2=d2,
        disc_r=math.exp(-r * T),
        disc_q=math.exp(-q * T),
        sqrt_t=sqrt_t,
    )


def price(spec: OptionSpec) -> float:
    r"""Black-Scholes-Merton fair value of the option described by ``spec``.

    Evaluates the unified formula
    :math:`V = \omega[S e^{-qT} N(\omega d_1) - K e^{-rT} N(\omega d_2)]`
    with :math:`\omega = +1` for a call and :math:`-1` for a put.
    """
    d = _compute_d(spec)
    w = spec.option_type.sign
    return w * (
        spec.spot * d.disc_q * _norm_cdf(w * d.d1)
        - spec.strike * d.disc_r * _norm_cdf(w * d.d2)
    )


def delta(spec: OptionSpec) -> float:
    r"""Sensitivity of price to spot, :math:`\partial V/\partial S`.

    :math:`\Delta_{\text{call}} = e^{-qT} N(d_1)`,
    :math:`\Delta_{\text{put}} = -e^{-qT} N(-d_1) = e^{-qT}(N(d_1) - 1)`.
    """
    d = _compute_d(spec)
    if spec.is_call:
        return d.disc_q * _norm_cdf(d.d1)
    return -d.disc_q * _norm_cdf(-d.d1)


def gamma(spec: OptionSpec) -> float:
    r"""Second derivative of price w.r.t. spot, :math:`\partial^2 V/\partial S^2`.

    :math:`\Gamma = e^{-qT}\,\phi(d_1)\,/\,(S\sigma\sqrt T)`; identical for
    calls and puts. Zero in the deterministic :math:`\sigma\sqrt T \to 0`
    limit (where :math:`d_1` is infinite and :math:`\phi(d_1)=0`).
    """
    d = _compute_d(spec)
    denom = spec.spot * spec.volatility * d.sqrt_t
    if denom < 1e-15 or math.isinf(d.d1):
        return 0.0
    return d.disc_q * _norm_pdf(d.d1) / denom


def vega(spec: OptionSpec) -> float:
    r"""Sensitivity of price to volatility, :math:`\partial V/\partial\sigma`.

    :math:`\mathcal{V} = S e^{-qT}\,\phi(d_1)\,\sqrt T`; identical for calls
    and puts. Reported per unit (absolute) change in volatility — divide by
    100 for the "per 1 vol point" convention.
    """
    d = _compute_d(spec)
    if math.isinf(d.d1):
        return 0.0
    return spec.spot * d.disc_q * _norm_pdf(d.d1) * d.sqrt_t


def theta(spec: OptionSpec) -> float:
    r"""Sensitivity of price to the passage of time, :math:`-\partial V/\partial T`.

    Returned as calendar-time decay (per year); a negative number means the
    option loses value as maturity approaches. With dividends,

    .. math::

        \Theta_{\text{call}} = -\frac{S e^{-qT}\phi(d_1)\sigma}{2\sqrt T}
            - r K e^{-rT} N(d_2) + q S e^{-qT} N(d_1),

    and the put analogue with the sign of the last two terms and the CDF
    arguments flipped.
    """
    d = _compute_d(spec)
    S, K, T = spec.spot, spec.strike, spec.maturity
    r, q, sigma = spec.rate, spec.dividend_yield, spec.volatility

    if math.isinf(d.d1):
        decay_term = 0.0
    else:
        decay_term = -S * d.disc_q * _norm_pdf(d.d1) * sigma / (2.0 * d.sqrt_t)

    if spec.is_call:
        return (
            decay_term
            - r * K * d.disc_r * _norm_cdf(d.d2)
            + q * S * d.disc_q * _norm_cdf(d.d1)
        )
    return (
        decay_term
        + r * K * d.disc_r * _norm_cdf(-d.d2)
        - q * S * d.disc_q * _norm_cdf(-d.d1)
    )


def rho(spec: OptionSpec) -> float:
    r"""Sensitivity of price to the risk-free rate, :math:`\partial V/\partial r`.

    :math:`\rho_{\text{call}} = K T e^{-rT} N(d_2)`,
    :math:`\rho_{\text{put}} = -K T e^{-rT} N(-d_2)`. Reported per unit
    change in :math:`r`.
    """
    d = _compute_d(spec)
    factor = spec.strike * spec.maturity * d.disc_r
    if spec.is_call:
        return factor * _norm_cdf(d.d2)
    return -factor * _norm_cdf(-d.d2)


def greeks(spec: OptionSpec) -> dict[str, float]:
    """Return all five closed-form Greeks in a single dictionary.

    Keys: ``delta``, ``gamma``, ``vega``, ``theta``, ``rho``.
    """
    return {
        "delta": delta(spec),
        "gamma": gamma(spec),
        "vega": vega(spec),
        "theta": theta(spec),
        "rho": rho(spec),
    }


def put_call_parity_rhs(spec: OptionSpec) -> float:
    r"""Right-hand side of put-call parity, :math:`C - P`.

    Parity states :math:`C - P = S_0 e^{-qT} - K e^{-rT}`. Exposed as a helper
    so tests (and users) can check the identity independently of which leg was
    priced.
    """
    return spec.spot * math.exp(-spec.dividend_yield * spec.maturity) - spec.strike * math.exp(
        -spec.rate * spec.maturity
    )
