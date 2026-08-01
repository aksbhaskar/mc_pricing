r"""Implied volatility solver via Brent's method.

The Black-Scholes price is strictly increasing in volatility (Vega
:math:`> 0`), so for any arbitrage-free market price there is a unique implied
volatility :math:`\sigma_{\text{imp}}` solving

.. math::

    V_{\text{BS}}(\sigma_{\text{imp}}) = V_{\text{market}}.

We solve this root-finding problem with Brent's method, which combines the
guaranteed convergence of bisection with the speed of inverse-quadratic
interpolation, robust and derivative-free, so it never diverges the way a
naive Newton iteration can when Vega is tiny (deep in/out of the money).
"""

from __future__ import annotations

from scipy.optimize import brentq

from . import black_scholes as bs
from .option import OptionSpec, OptionType


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float = 0.0,
    *,
    vol_lower: float = 1e-6,
    vol_upper: float = 5.0,
    tol: float = 1e-8,
) -> float:
    r"""Solve for the Black-Scholes implied volatility of an observed price.

    Parameters
    ----------
    price:
        Observed market price of the option.
    spot, strike, maturity, rate, option_type, dividend_yield:
        The remaining Black-Scholes inputs (everything except volatility).
    vol_lower, vol_upper:
        Bracketing interval for the search, in volatility units. The default
        ``[1e-6, 5.0]`` spans essentially every traded regime.
    tol:
        Absolute tolerance on the volatility root.

    Returns
    -------
    float
        The implied volatility :math:`\sigma_{\text{imp}}`.

    Raises
    ------
    ValueError
        If ``price`` violates the no-arbitrage bounds (below discounted
        intrinsic value or above the underlying's forward value), so that no
        root exists in the bracket.
    """

    def objective(sigma: float) -> float:
        spec = OptionSpec(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=sigma,
            option_type=OptionType(option_type),
            dividend_yield=dividend_yield,
        )
        return bs.price(spec) - price

    f_lo = objective(vol_lower)
    f_hi = objective(vol_upper)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            "price is outside the no-arbitrage range for the given bracket; "
            f"BS price at sigma={vol_lower} is {f_lo + price:.6f}, "
            f"at sigma={vol_upper} is {f_hi + price:.6f}, target is {price:.6f}"
        )

    return float(brentq(objective, vol_lower, vol_upper, xtol=tol))
