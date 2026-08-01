"""mc_pricing: a Monte Carlo option-pricing engine for vanilla European options.

The package is organised in layers, each in its own module:

* :mod:`mc_pricing.option`: the :class:`OptionSpec` value object and payoff type.
* :mod:`mc_pricing.black_scholes`: closed-form Black-Scholes-Merton prices and
  Greeks (the analytical ground truth).
* :mod:`mc_pricing.engine`: exact geometric-Brownian-motion terminal sampling.
* :mod:`mc_pricing.payoff`: vanilla call/put payoff functions.
* :mod:`mc_pricing.variance_reduction`: antithetic and control variates.
* :mod:`mc_pricing.pricer`: the user-facing :func:`monte_carlo_price`.
* :mod:`mc_pricing.greeks`: pathwise and finite-difference Monte Carlo Greeks.
* :mod:`mc_pricing.stats`: confidence intervals and estimate summaries.
* :mod:`mc_pricing.implied_vol`: Brent's-method implied-volatility solver.

Quick start
-----------
>>> from mc_pricing import OptionSpec, OptionType, monte_carlo_price, black_scholes
>>> spec = OptionSpec(spot=100, strike=100, maturity=1.0, rate=0.05,
...                   volatility=0.2, option_type=OptionType.CALL)
>>> analytic = black_scholes.price(spec)
>>> mc = monte_carlo_price(spec, n_paths=200_000, antithetic=True,
...                        control_variate=True, seed=0)
>>> abs(mc.price - analytic) < 3 * mc.std_error
True
"""

from __future__ import annotations

from . import black_scholes, greeks
from .greeks import GreekEstimate, monte_carlo_greeks
from .implied_vol import implied_volatility
from .option import OptionSpec, OptionType
from .pricer import PricingResult, monte_carlo_price
from .stats import MCResult, summarize

__all__ = [
    "OptionSpec",
    "OptionType",
    "MCResult",
    "summarize",
    "PricingResult",
    "monte_carlo_price",
    "GreekEstimate",
    "monte_carlo_greeks",
    "implied_volatility",
    "black_scholes",
    "greeks",
]

__version__ = "0.1.0"
