r"""Vanilla European payoff functions.

Kept in their own tiny module because both the pricer and the pathwise Greek
estimators evaluate payoffs, and factoring them out avoids duplicating the
:math:`\max(\cdot, 0)` logic (and its sign convention) in several places.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .option import OptionSpec


def terminal_payoff(
    spec: OptionSpec, terminal_prices: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Undiscounted payoff :math:`\max(\omega(S_T - K), 0)` for each path.

    :math:`\omega = +1` for a call, :math:`-1` for a put. Vectorised over the
    array of terminal prices.
    """
    w = spec.option_type.sign
    return np.maximum(w * (terminal_prices - spec.strike), 0.0)
