r"""The user-facing Monte Carlo pricer.

This module ties the engine, payoff, variance-reduction and statistics layers
together into a single :func:`monte_carlo_price` entry point. It prices a
vanilla European option as the discounted risk-neutral expectation of its
payoff,

.. math::

    V = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\big[\max(\omega(S_T - K), 0)\big],

estimated by simulating :math:`S_T` and averaging, optionally with antithetic
and/or control variates. Every call returns not just a price but its standard
error, confidence interval, and the variance-reduction ratio actually achieved
relative to plain Monte Carlo on the *same* random draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import variance_reduction as vr
from .engine import simulate_terminal
from .option import OptionSpec
from .payoff import terminal_payoff
from .stats import MCResult, summarize


@dataclass(frozen=True, slots=True)
class PricingResult:
    """The full result of a Monte Carlo pricing run.

    Attributes
    ----------
    result:
        The :class:`~mc_pricing.stats.MCResult`: price estimate, standard
        error, and confidence interval.
    antithetic:
        Whether antithetic variates were used.
    control_variate:
        Whether the terminal-price control variate was used.
    variance_reduction_ratio:
        Ratio of the *plain* estimator's mean-variance (SE²) to this
        estimator's, computed on the same simulated paths. ``> 1`` means
        variance was reduced; e.g. ``4.0`` means plain MC would need ~4× the
        paths to match this accuracy. Exactly ``1.0`` for a plain run.
    control_beta:
        The fitted control-variate coefficient :math:`b^\\star` (``0`` if the
        control variate was not used).
    n_paths_simulated:
        Total number of terminal prices actually simulated (may be one more
        than requested when antithetic pairing rounds an odd count up).
    """

    result: MCResult
    antithetic: bool
    control_variate: bool
    variance_reduction_ratio: float
    control_beta: float
    n_paths_simulated: int

    @property
    def price(self) -> float:
        """Shorthand for the point-estimate price."""
        return self.result.estimate

    @property
    def std_error(self) -> float:
        """Shorthand for the estimate's standard error."""
        return self.result.std_error

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Shorthand for the estimate's confidence interval."""
        return self.result.confidence_interval


def monte_carlo_price(
    spec: OptionSpec,
    n_paths: int = 100_000,
    *,
    antithetic: bool = False,
    control_variate: bool = False,
    confidence: float = 0.95,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> PricingResult:
    r"""Price a European option by Monte Carlo simulation.

    Parameters
    ----------
    spec:
        The option and market specification.
    n_paths:
        Number of terminal prices to simulate. With ``antithetic=True`` this is
        rounded up to the next even number so pairs are balanced.
    antithetic:
        Enable antithetic variates (mirror each normal :math:`Z` with
        :math:`-Z`).
    control_variate:
        Enable the terminal-price control variate.
    confidence:
        Confidence level for the reported interval (default 95%).
    seed:
        Convenience seed used to build a default RNG when ``rng`` is not given.
    rng:
        An explicit NumPy generator (takes precedence over ``seed``); inject
        this to share a random stream across calls.

    Returns
    -------
    PricingResult
        Price estimate with uncertainty and variance-reduction diagnostics.

    Notes
    -----
    The estimator is the discounted sample mean of the payoff. When variance
    reduction is enabled the per-sample estimator is transformed (paired and/or
    control-adjusted) but its expectation is unchanged, so the price stays
    unbiased while the confidence interval tightens.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    discount = np.exp(-spec.rate * spec.maturity)

    if antithetic:
        # Round up to an even count so every draw has its mirror.
        n_eff = n_paths + (n_paths % 2)
    else:
        n_eff = n_paths

    sample = simulate_terminal(spec, n_eff, rng, antithetic=antithetic)
    terminal = sample.terminal_prices
    disc_payoffs = discount * terminal_payoff(spec, terminal)

    # Plain (no variance reduction) baseline on the *same* paths, for an
    # honest reduction ratio. This is just the mean of the discounted payoffs.
    plain = summarize(disc_payoffs, confidence=confidence)
    plain_mean_var = plain.std_error**2  # SE² of plain estimator

    # Build the per-sample estimator for the requested technique(s).
    beta = 0.0
    if antithetic and control_variate:
        # Collapse pairs first, then control on the paired estimators.
        paired_payoffs = vr.collapse_antithetic(disc_payoffs)
        paired_controls = vr.collapse_antithetic(terminal)
        cv = vr.apply_control_variate(
            paired_payoffs, paired_controls, vr.expected_terminal(spec)
        )
        per_sample = cv.adjusted
        beta = cv.beta
    elif antithetic:
        per_sample = vr.collapse_antithetic(disc_payoffs)
    elif control_variate:
        cv = vr.apply_control_variate(
            disc_payoffs, terminal, vr.expected_terminal(spec)
        )
        per_sample = cv.adjusted
        beta = cv.beta
    else:
        per_sample = disc_payoffs

    result = summarize(per_sample, confidence=confidence)

    reduced_mean_var = result.std_error**2
    if reduced_mean_var > 0.0:
        ratio = plain_mean_var / reduced_mean_var
    else:
        ratio = float("inf")

    return PricingResult(
        result=result,
        antithetic=antithetic,
        control_variate=control_variate,
        variance_reduction_ratio=ratio,
        control_beta=beta,
        n_paths_simulated=int(terminal.size),
    )
