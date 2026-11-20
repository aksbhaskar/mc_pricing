r"""Variance-reduction techniques for Monte Carlo option pricing.

Two complementary techniques are implemented, both of which leave the estimator
**unbiased** while shrinking its variance, meaning the confidence interval
narrows for the same number of paths (equivalently, fewer paths are needed for
a target accuracy).

Antithetic variates
--------------------
For each standard-normal draw :math:`Z` we also use its mirror :math:`-Z`. The
paired estimator is the average of the two discounted payoffs. Since the payoff
is a monotone function of :math:`Z`, the pair is negatively correlated and

.. math::

    \operatorname{Var}\!\left[\tfrac12(Y(Z) + Y(-Z))\right]
    = \tfrac12\operatorname{Var}[Y]\,(1 + \rho) \le \tfrac12\operatorname{Var}[Y],

with :math:`\rho = \operatorname{Corr}(Y(Z), Y(-Z)) < 0`. The pairing happens
in :mod:`mc_pricing.engine` (via mirrored normals); this module only collapses
paired samples into per-pair estimators.

Control variates
----------------
We exploit a correlated quantity whose expectation is known in closed form.
The terminal price :math:`S_T` is the natural control because its risk-neutral
mean is exactly :math:`\mathbb{E}[S_T] = S_0 e^{(r-q)T}`. Given discounted
payoffs :math:`Y` and controls :math:`X = S_T`, the controlled estimator is

.. math::

    Y^\star_i = Y_i - b\,(X_i - \mathbb{E}[X]),

which is unbiased for any :math:`b`. The variance-minimising coefficient is

.. math::

    b^\star = \frac{\operatorname{Cov}(Y, X)}{\operatorname{Var}(X)},

estimated from the sample. The resulting variance is
:math:`\operatorname{Var}[Y](1 - \rho_{YX}^2)`, so the reduction is dictated by
how strongly the payoff correlates with the terminal price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .option import OptionSpec
from .stats import CentredMoments


def collapse_antithetic(values: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Average antithetic pairs into per-pair estimators.

    ``values`` must be laid out as ``[f(Z_1), …, f(Z_m), f(-Z_1), …, f(-Z_m)]``
    (the ordering produced by :func:`mc_pricing.engine.draw_normals`). The
    returned array of length :math:`m` holds
    :math:`\tfrac12(f(Z_i) + f(-Z_i))`, each element being one *independent*
    sample for the purposes of the CLT.
    """
    n = values.size
    if n % 2 != 0:
        raise ValueError(
            f"antithetic values must have even length, got {n}"
        )
    half = n // 2
    return 0.5 * (values[:half] + values[half:])


def expected_terminal(spec: OptionSpec) -> float:
    r"""Closed-form risk-neutral mean of the terminal price,
    :math:`\mathbb{E}[S_T] = S_0 e^{(r-q)T}`.
    """
    return spec.spot * math.exp((spec.rate - spec.dividend_yield) * spec.maturity)


@dataclass(frozen=True, slots=True)
class ControlVariateResult:
    """Per-sample controlled estimator values and the fitted coefficient.

    Attributes
    ----------
    adjusted:
        The controlled per-sample estimator values :math:`Y^\\star_i`, ready to
        be handed to :func:`mc_pricing.stats.summarize`.
    beta:
        The fitted optimal coefficient :math:`b^\\star`.
    """

    adjusted: NDArray[np.float64]
    beta: float


def apply_control_variate(
    discounted_payoffs: NDArray[np.float64],
    controls: NDArray[np.float64],
    control_mean: float,
) -> ControlVariateResult:
    r"""Apply the terminal-price control variate to discounted payoffs.

    Estimates :math:`b^\star = \widehat{\operatorname{Cov}}(Y, X) /
    \widehat{\operatorname{Var}}(X)` from the sample and returns the adjusted
    estimator :math:`Y - b^\star(X - \mathbb{E}[X])`.

    Notes
    -----
    Using an in-sample :math:`b^\star` introduces a technically :math:`O(1/n)`
    bias (the coefficient is correlated with the samples). It is negligible at
    the path counts used here and is standard practice; a fully rigorous
    implementation would fit :math:`b^\star` on an independent pilot run.
    """
    y = np.asarray(discounted_payoffs, dtype=np.float64)
    x = np.asarray(controls, dtype=np.float64)
    n = x.size
    xc = x - x.mean()
    m2_xx = float(np.dot(xc, xc))
    var_x = m2_xx / (n - 1) if n > 1 else 0.0
    if var_x < 1e-30:
        # Degenerate control (e.g. sigma = 0 → constant S_T): no adjustment.
        return ControlVariateResult(adjusted=y.copy(), beta=0.0)
    beta = float(np.dot(y - y.mean(), xc)) / m2_xx
    adjusted = y - beta * (x - control_mean)
    return ControlVariateResult(adjusted=adjusted, beta=beta)


def control_variate_moments(
    moments: CentredMoments, control_mean: float
) -> tuple[float, float, float]:
    r"""Controlled estimator statistics straight from the sample moments.

    Equivalent to :func:`apply_control_variate` followed by summarising the
    adjusted samples, but without materialising them: with
    :math:`b^\star = M_{xy}/M_{xx}`,

    .. math::

        \bar Y^\star = \bar Y - b^\star(\bar X - \mathbb{E}[X]),\qquad
        M^\star_2 = M_{yy} - 2b^\star M_{xy} + (b^\star)^2 M_{xx}.

    Returns
    -------
    tuple[float, float, float]
        ``(mean, m2, beta)`` of the controlled per-sample estimator.
    """
    n = moments.n
    var_x = moments.m2_xx / (n - 1) if n > 1 else 0.0
    if var_x < 1e-30:
        return moments.mean_y, moments.m2_yy, 0.0
    beta = moments.m2_xy / moments.m2_xx
    mean = moments.mean_y - beta * (moments.mean_x - control_mean)
    m2 = moments.m2_yy - 2.0 * beta * moments.m2_xy + beta * beta * moments.m2_xx
    return mean, max(m2, 0.0), beta
