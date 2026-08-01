r"""The Monte Carlo simulation engine: exact GBM terminal sampling.

For a *European* payoff the option value depends only on the terminal price
:math:`S_T`, never on the path in between. There is therefore no need for
Euler/Milstein path discretisation — we sample :math:`S_T` **exactly** from its
known log-normal law,

.. math::

    S_T = S_0 \exp\!\Big[(r - q - \tfrac12\sigma^2)T + \sigma\sqrt T\,Z\Big],
    \qquad Z \sim \mathcal{N}(0, 1).

This is both faster and bias-free (a discretised path would introduce
time-stepping bias). The engine's sole job is to turn standard-normal draws
:math:`Z` into terminal prices; the antithetic construction and the mapping to
payoffs live in higher layers so that variance-reduction and Greek estimators
can all reuse the very same normals (common random numbers).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .option import OptionSpec


@dataclass(frozen=True, slots=True)
class TerminalSample:
    r"""A batch of simulated terminal prices and the normals that generated them.

    Retaining the driving normals :math:`Z` alongside :math:`S_T` is what makes
    pathwise Greek estimators possible: the Vega estimator, for instance, needs
    :math:`\partial S_T/\partial\sigma = S_T(\sqrt T\,Z - \sigma T)`, which is
    expressed directly in terms of these :math:`Z` values.

    Attributes
    ----------
    terminal_prices:
        Array of simulated :math:`S_T` values.
    normals:
        The standard-normal draws :math:`Z` (including any antithetic mirror
        images), aligned element-wise with ``terminal_prices``.
    antithetic:
        Whether the sample was generated with antithetic variates.
    """

    terminal_prices: NDArray[np.float64]
    normals: NDArray[np.float64]
    antithetic: bool


def draw_normals(
    n_paths: int,
    rng: np.random.Generator,
    antithetic: bool = False,
) -> NDArray[np.float64]:
    r"""Draw the standard-normal shocks :math:`Z` driving the simulation.

    Parameters
    ----------
    n_paths:
        Total number of terminal samples requested.
    rng:
        A NumPy :class:`~numpy.random.Generator` (injected so simulations are
        reproducible and independent streams can be composed).
    antithetic:
        If ``True``, only :math:`\lceil n/2\rceil` independent normals are
        drawn and each is paired with its negation :math:`-Z`. Because
        :math:`Z` and :math:`-Z` are perfectly negatively correlated, the
        average of a monotone function over the pair has lower variance than
        two independent draws. When ``n_paths`` is odd the array is trimmed
        back to exactly ``n_paths`` after mirroring.

    Returns
    -------
    numpy.ndarray
        The :math:`Z` draws. For the antithetic case the first half are the
        independent draws and the second half their mirror images, so callers
        can reshape into pairs if needed.
    """
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")

    if not antithetic:
        return rng.standard_normal(n_paths)

    half = (n_paths + 1) // 2
    base = rng.standard_normal(half)
    z = np.concatenate([base, -base])
    return z[:n_paths]


def simulate_terminal(
    spec: OptionSpec,
    n_paths: int,
    rng: np.random.Generator,
    antithetic: bool = False,
    normals: NDArray[np.float64] | None = None,
) -> TerminalSample:
    r"""Simulate terminal underlying prices :math:`S_T` under the risk-neutral GBM.

    Applies the exact log-normal solution of the GBM SDE to a set of
    standard-normal shocks. Supplying ``normals`` lets a caller reuse an
    existing set of draws — the basis of *common random numbers*, essential for
    low-variance finite-difference Greeks where the bumped and un-bumped prices
    must share the same randomness.

    Parameters
    ----------
    spec:
        The option/market specification providing :math:`S_0, r, q, \sigma, T`.
    n_paths:
        Number of terminal prices to simulate (ignored if ``normals`` given).
    rng:
        Random generator used only when ``normals`` is ``None``.
    antithetic:
        Whether to use antithetic sampling when drawing fresh normals.
    normals:
        Optional pre-drawn standard normals to reuse. If provided,
        ``n_paths`` and ``antithetic`` are taken from its shape/flag.

    Returns
    -------
    TerminalSample
        The simulated terminal prices together with their driving normals.
    """
    if normals is None:
        z = draw_normals(n_paths, rng, antithetic=antithetic)
    else:
        z = np.asarray(normals, dtype=np.float64).ravel()

    drift = (spec.rate - spec.dividend_yield - 0.5 * spec.volatility**2) * spec.maturity
    diffusion = spec.volatility * np.sqrt(spec.maturity) * z
    terminal = spec.spot * np.exp(drift + diffusion)

    return TerminalSample(terminal_prices=terminal, normals=z, antithetic=antithetic)
