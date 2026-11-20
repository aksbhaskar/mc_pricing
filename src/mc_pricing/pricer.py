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

import math
import warnings
from dataclasses import dataclass

import numpy as np

from . import variance_reduction as vr
from .engine import simulate_terminal
from .option import OptionSpec
from .payoff import terminal_payoff
from .stats import CentredMoments, MCResult, summarize_moments


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
    chunk_size: int | None = None,
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
    chunk_size:
        If given, simulate at most this many paths at a time (rounded up to an
        even count with ``antithetic=True`` so pairs never straddle chunks) and
        merge their centred moments, bounding memory to ``O(chunk_size)``. The
        random stream and the estimator are unchanged; results agree with the
        unchunked run up to floating-point rounding.

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
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if rng is None:
        rng = np.random.default_rng(seed)

    discount = np.exp(-spec.rate * spec.maturity)

    if antithetic:
        # Round up to an even count so every draw has its mirror.
        n_eff = n_paths + (n_paths % 2)
    else:
        n_eff = n_paths

    chunk = n_eff if chunk_size is None else chunk_size
    if antithetic:
        chunk += chunk % 2

    # ``est`` holds the moments of the per-sample estimator (paired when
    # antithetic) and, with a control variate, of the matching controls.
    # ``plain`` holds the moments of the raw discounted payoffs on the *same*
    # paths, for an honest reduction ratio; without antithetic pairing it is
    # the same sample as ``est``.
    est: CentredMoments | None = None
    plain: CentredMoments | None = None
    remaining = n_eff
    while remaining > 0:
        m = min(chunk, remaining)
        remaining -= m
        terminal = simulate_terminal(spec, m, rng, antithetic=antithetic).terminal_prices
        disc_payoffs = terminal_payoff(spec, terminal)
        disc_payoffs *= discount
        if antithetic:
            # Collapse pairs first, then (optionally) control on the pairs.
            batch_plain = CentredMoments.from_arrays(disc_payoffs)
            plain = batch_plain if plain is None else plain.merge(batch_plain)
            batch = CentredMoments.from_arrays(
                vr.collapse_antithetic(disc_payoffs),
                vr.collapse_antithetic(terminal) if control_variate else None,
            )
        else:
            batch = CentredMoments.from_arrays(
                disc_payoffs, terminal if control_variate else None
            )
        est = batch if est is None else est.merge(batch)

    assert est is not None
    if plain is None:
        plain = est

    plain_result = summarize_moments(plain.n, plain.mean_y, plain.m2_yy, confidence)

    beta = 0.0
    if control_variate:
        mean, m2, beta = vr.control_variate_moments(est, vr.expected_terminal(spec))
        result = summarize_moments(est.n, mean, m2, confidence)
    elif antithetic:
        result = summarize_moments(est.n, est.mean_y, est.m2_yy, confidence)
    else:
        result = plain_result

    plain_mean_var = plain_result.std_error**2  # SE² of plain estimator
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
        n_paths_simulated=int(n_eff),
    )


def monte_carlo_price_to_accuracy(
    spec: OptionSpec,
    target_std_error: float,
    *,
    antithetic: bool = True,
    control_variate: bool = True,
    pilot_paths: int = 20_000,
    max_paths: int = 50_000_000,
    confidence: float = 0.95,
    chunk_size: int | None = 1_000_000,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> PricingResult:
    r"""Price to a target standard error, sizing the run from a pilot.

    Variance reduction buys accuracy, not speed, at a fixed path count; this
    entry point turns that into speed. A pilot run of ``pilot_paths`` estimates
    the per-sample standard deviation :math:`s` of the chosen estimator (by
    default antithetic + control variate), the number of samples needed for
    :math:`s/\sqrt n \le` ``target_std_error`` is computed, and a **fresh,
    independent** final run of that size is priced and returned. The pilot is
    discarded, so the final estimate, its interval and its fitted control
    coefficient are not conditioned on the sizing step.

    Parameters
    ----------
    spec:
        The option and market specification.
    target_std_error:
        Desired standard error of the final price estimate.
    antithetic, control_variate:
        Estimator to use for both the pilot and the final run. Pass ``False``
        for both to size a plain Monte Carlo run for comparison.
    pilot_paths:
        Paths in the pilot run used only to estimate the variance.
    max_paths:
        Upper bound on the final run's paths; a :class:`UserWarning` is issued
        and the bound used if the target would need more.
    confidence:
        Confidence level for the reported interval.
    chunk_size:
        Passed to :func:`monte_carlo_price` for the final run to bound memory.
    seed, rng:
        As for :func:`monte_carlo_price`. The pilot and the final run consume
        consecutive, non-overlapping draws from the same generator.

    Returns
    -------
    PricingResult
        The final run's result. Its standard error is close to, but being
        itself an estimate not guaranteed below, ``target_std_error``.
        ``n_paths_simulated`` excludes the pilot's paths.
    """
    if not target_std_error > 0.0:
        raise ValueError(f"target_std_error must be positive, got {target_std_error}")
    if rng is None:
        rng = np.random.default_rng(seed)

    pilot = monte_carlo_price(
        spec,
        pilot_paths,
        antithetic=antithetic,
        control_variate=control_variate,
        confidence=confidence,
        rng=rng,
    )
    paths_per_sample = 2 if antithetic else 1
    n_samples = max(2, math.ceil((pilot.result.sample_std / target_std_error) ** 2))
    n_paths = n_samples * paths_per_sample
    if n_paths > max_paths:
        warnings.warn(
            f"target_std_error={target_std_error:g} needs ~{n_paths:,} paths; "
            f"capping at max_paths={max_paths:,}",
            UserWarning,
            stacklevel=2,
        )
        n_paths = max_paths

    return monte_carlo_price(
        spec,
        n_paths,
        antithetic=antithetic,
        control_variate=control_variate,
        confidence=confidence,
        rng=rng,
        chunk_size=chunk_size,
    )
