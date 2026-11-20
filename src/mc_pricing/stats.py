"""Statistical utilities for Monte Carlo estimation.

Every Monte Carlo estimate in this package is reported together with a
standard error and a confidence interval, because a point estimate without an
error bar is meaningless in a stochastic simulation. This module provides the
small container that carries those quantities and the CLT machinery that
produces them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class MCResult:
    r"""The outcome of a Monte Carlo estimate with its uncertainty.

    Attributes
    ----------
    estimate:
        The sample-mean point estimate :math:`\hat\mu`.
    std_error:
        The standard error of the mean,
        :math:`\mathrm{SE} = s / \sqrt{n}`, where :math:`s` is the sample
        standard deviation and :math:`n` the number of samples.
    n_samples:
        The number of independent samples :math:`n` used. For antithetic
        estimators this counts *pairs*, since a pair is one independent draw.
    confidence:
        The nominal coverage of :attr:`confidence_interval` (e.g. ``0.95``).
    sample_std:
        The sample standard deviation :math:`s` of the per-sample estimator.
        Exposed so callers can compute variance-reduction ratios directly.
    """

    estimate: float
    std_error: float
    n_samples: int
    confidence: float
    sample_std: float

    @property
    def confidence_interval(self) -> tuple[float, float]:
        r"""The two-sided CLT confidence interval for the mean.

        By the Central Limit Theorem the sample mean is asymptotically normal,
        so an approximate :math:`(1-\alpha)` interval is

        .. math::

            \hat\mu \pm z_{1-\alpha/2}\,\frac{s}{\sqrt n},

        with :math:`z` the standard-normal quantile. We use the normal (rather
        than Student-t) quantile because Monte Carlo sample sizes are large
        enough that the difference is negligible.
        """
        z = _normal_quantile(0.5 + self.confidence / 2.0)
        half = z * self.std_error
        return (self.estimate - half, self.estimate + half)

    @property
    def variance(self) -> float:
        """Per-sample variance :math:`s^2` of the estimator."""
        return self.sample_std**2

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        lo, hi = self.confidence_interval
        pct = int(round(self.confidence * 100))
        return (
            f"MCResult(estimate={self.estimate:.6f}, "
            f"se={self.std_error:.2e}, "
            f"{pct}%CI=[{lo:.6f}, {hi:.6f}], n={self.n_samples})"
        )


def summarize(
    samples: NDArray[np.float64], confidence: float = 0.95
) -> MCResult:
    r"""Summarise an array of i.i.d. estimator samples into an :class:`MCResult`.

    Given samples :math:`X_1,\dots,X_n` of an unbiased estimator, this returns
    the sample mean as the point estimate and the standard error
    :math:`s/\sqrt n` computed with the unbiased (:math:`n-1`) variance.

    Parameters
    ----------
    samples:
        1-D array of per-sample estimator values (e.g. discounted payoffs).
    confidence:
        Nominal confidence level for the reported interval.

    Raises
    ------
    ValueError
        If ``samples`` is empty or ``confidence`` is not in ``(0, 1)``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    samples = np.asarray(samples, dtype=np.float64).ravel()
    n = samples.size
    if n == 0:
        raise ValueError("cannot summarize an empty sample array")

    mean = float(samples.mean())
    # ddof=1 → unbiased sample variance. For n == 1 the SE is undefined; report
    # zero std/SE rather than a NaN so the pipeline degrades gracefully.
    std = float(samples.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n)
    return MCResult(
        estimate=mean,
        std_error=se,
        n_samples=n,
        confidence=confidence,
        sample_std=std,
    )


def summarize_moments(
    n: int, mean: float, m2: float, confidence: float = 0.95
) -> MCResult:
    r"""Build an :class:`MCResult` from a sample size, mean and centred sum of
    squares :math:`M_2 = \sum_i (X_i - \bar X)^2`.

    Gives the same statistics as :func:`summarize` (unbiased ``n - 1``
    variance) without needing the samples themselves.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n <= 0:
        raise ValueError("cannot summarize an empty sample")
    std = math.sqrt(max(m2, 0.0) / (n - 1)) if n > 1 else 0.0
    return MCResult(
        estimate=float(mean),
        std_error=std / math.sqrt(n),
        n_samples=int(n),
        confidence=confidence,
        sample_std=std,
    )


@dataclass(frozen=True, slots=True)
class CentredMoments:
    r"""Running centred moments of paired samples :math:`(Y_i, X_i)`.

    Holds the count, the two means and the centred sums
    :math:`M_{yy} = \sum (Y-\bar Y)^2`, :math:`M_{xx}` and
    :math:`M_{xy} = \sum (Y-\bar Y)(X-\bar X)`. Batches are combined with
    Chan et al.'s pairwise update, which stays cancellation-resistant (unlike
    raw sums of squares), so a simulation can be streamed in chunks and still
    give the variances and covariance of the whole sample. When no ``x`` is
    supplied the ``x`` fields stay zero.
    """

    n: int
    mean_y: float
    mean_x: float
    m2_yy: float
    m2_xx: float
    m2_xy: float

    @classmethod
    def from_arrays(
        cls, y: NDArray[np.float64], x: NDArray[np.float64] | None = None
    ) -> "CentredMoments":
        y = np.asarray(y, dtype=np.float64).ravel()
        n = y.size
        if n == 0:
            raise ValueError("cannot take moments of an empty sample")
        mean_y = float(y.mean())
        yc = y - mean_y
        m2_yy = float(np.dot(yc, yc))
        if x is None:
            return cls(n, mean_y, 0.0, m2_yy, 0.0, 0.0)
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != n:
            raise ValueError("y and x must have the same length")
        mean_x = float(x.mean())
        xc = x - mean_x
        return cls(
            n, mean_y, mean_x, m2_yy, float(np.dot(xc, xc)), float(np.dot(yc, xc))
        )

    def merge(self, other: "CentredMoments") -> "CentredMoments":
        n = self.n + other.n
        dy = other.mean_y - self.mean_y
        dx = other.mean_x - self.mean_x
        w = self.n * other.n / n
        return CentredMoments(
            n=n,
            mean_y=self.mean_y + dy * other.n / n,
            mean_x=self.mean_x + dx * other.n / n,
            m2_yy=self.m2_yy + other.m2_yy + dy * dy * w,
            m2_xx=self.m2_xx + other.m2_xx + dx * dx * w,
            m2_xy=self.m2_xy + other.m2_xy + dy * dx * w,
        )


def _normal_quantile(p: float) -> float:
    r"""Inverse standard-normal CDF (the probit function) :math:`\Phi^{-1}(p)`.

    Implemented with Acklam's rational approximation, which is accurate to
    roughly 1e-9 across the whole open interval, more than enough for
    confidence-interval half-widths and avoids a hard SciPy dependency in this
    otherwise NumPy-only module.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"quantile argument must be in (0, 1), got {p}")

    # Coefficients for Acklam's algorithm.
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        u = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]) / (
            (((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1.0
        )
    if p <= p_high:
        u = p - 0.5
        t = u * u
        return (((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t + a[5]) * u / (
            (((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t + b[4]) * t + 1.0)
        )
    u = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        ((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]
    ) / ((((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1.0)
