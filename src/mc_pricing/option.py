"""Option specification and market data containers.

This module defines the immutable value objects that describe a vanilla
European option and the market environment it lives in. Keeping these as
frozen dataclasses means an :class:`OptionSpec` can be passed freely between
the analytical, Monte Carlo, and Greek engines without any risk of a downstream
routine mutating shared state.

Notation used throughout the package
------------------------------------
============  ====================================================
Symbol        Meaning
============  ====================================================
``S`` / S0    Current (spot) price of the underlying asset
``K``         Strike price
``T``         Time to maturity, in years
``r``         Continuously-compounded risk-free rate
``q``         Continuous dividend yield of the underlying
``sigma`` / σ Volatility of the underlying's log-returns (annualised)
============  ====================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OptionType(str, Enum):
    """The two vanilla payoff types supported by the library.

    Inheriting from :class:`str` makes the enum trivially serialisable and
    lets callers write ``OptionType("call")`` or compare directly against the
    string ``"call"``, convenient when specs are built from config files.
    """

    CALL = "call"
    PUT = "put"

    @property
    def sign(self) -> int:
        r"""Return ``+1`` for a call and ``-1`` for a put.

        This is the payoff sign :math:`\omega` appearing in the unified
        Black-Scholes formula

        .. math::

            V = \omega\left[S e^{-qT} N(\omega d_1) - K e^{-rT} N(\omega d_2)\right],

        which collapses the call and put cases into a single expression.
        """
        return 1 if self is OptionType.CALL else -1


@dataclass(frozen=True, slots=True)
class OptionSpec:
    """An immutable description of a vanilla European option and its market.

    Parameters
    ----------
    spot:
        Current price :math:`S_0` of the underlying asset. Must be positive.
    strike:
        Strike price :math:`K`. Must be positive.
    maturity:
        Time to expiry :math:`T` in years. Must be positive.
    rate:
        Continuously-compounded annual risk-free rate :math:`r`.
    volatility:
        Annualised volatility :math:`\\sigma` of the underlying's
        log-returns. Must be non-negative.
    option_type:
        Whether the contract is a call or a put.
    dividend_yield:
        Continuous dividend yield :math:`q` of the underlying. Defaults to
        zero (non-dividend-paying asset).

    Notes
    -----
    The payoff at maturity is :math:`\\max(\\omega(S_T - K), 0)` where
    :math:`\\omega = \\pm 1` is :attr:`OptionType.sign`. Under the
    risk-neutral measure the fair value today is the discounted expectation
    :math:`e^{-rT}\\,\\mathbb{E}^{\\mathbb{Q}}[\\text{payoff}]`, which every
    pricer in this package estimates or evaluates in closed form.
    """

    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    option_type: OptionType = OptionType.CALL
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        # Coerce a string option_type (e.g. loaded from JSON) into the enum.
        if not isinstance(self.option_type, OptionType):
            object.__setattr__(self, "option_type", OptionType(self.option_type))

        if self.spot <= 0.0:
            raise ValueError(f"spot must be positive, got {self.spot}")
        if self.strike <= 0.0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.maturity <= 0.0:
            raise ValueError(f"maturity must be positive, got {self.maturity}")
        if self.volatility < 0.0:
            raise ValueError(f"volatility must be non-negative, got {self.volatility}")

    @property
    def is_call(self) -> bool:
        """``True`` if this spec describes a call option."""
        return self.option_type is OptionType.CALL

    def with_spot(self, spot: float) -> "OptionSpec":
        """Return a copy of this spec with a new :attr:`spot`.

        Used pervasively by the bump-and-reprice Greek estimators, which need
        to perturb a single parameter while holding everything else fixed.
        """
        return self._replace(spot=spot)

    def with_volatility(self, volatility: float) -> "OptionSpec":
        """Return a copy of this spec with a new :attr:`volatility`."""
        return self._replace(volatility=volatility)

    def with_rate(self, rate: float) -> "OptionSpec":
        """Return a copy of this spec with a new :attr:`rate`."""
        return self._replace(rate=rate)

    def with_maturity(self, maturity: float) -> "OptionSpec":
        """Return a copy of this spec with a new :attr:`maturity`."""
        return self._replace(maturity=maturity)

    def _replace(self, **changes: object) -> "OptionSpec":
        """Functional update helper (frozen dataclasses forbid attribute writes)."""
        fields = {
            "spot": self.spot,
            "strike": self.strike,
            "maturity": self.maturity,
            "rate": self.rate,
            "volatility": self.volatility,
            "option_type": self.option_type,
            "dividend_yield": self.dividend_yield,
        }
        fields.update(changes)
        return OptionSpec(**fields)  # type: ignore[arg-type]
