"""Tests for the confidence-interval / summary statistics layer."""

from __future__ import annotations

import numpy as np
import pytest

from mc_pricing.stats import MCResult, summarize
from mc_pricing.stats import _normal_quantile


def test_summarize_basic():
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res = summarize(samples)
    assert res.estimate == pytest.approx(3.0)
    assert res.n_samples == 5
    # sample std of 1..5 (ddof=1) is sqrt(2.5)
    assert res.sample_std == pytest.approx(np.sqrt(2.5))


def test_confidence_interval_widens_with_level():
    samples = np.random.default_rng(0).standard_normal(10_000)
    r90 = summarize(samples, 0.90)
    r99 = summarize(samples, 0.99)
    w90 = r90.confidence_interval[1] - r90.confidence_interval[0]
    w99 = r99.confidence_interval[1] - r99.confidence_interval[0]
    assert w99 > w90


def test_ci_contains_true_mean_for_normal():
    """A 95% CI for the mean of standard normals should contain 0 here."""
    samples = np.random.default_rng(1).standard_normal(100_000)
    res = summarize(samples, 0.95)
    lo, hi = res.confidence_interval
    assert lo <= 0.0 <= hi


def test_normal_quantile_known_values():
    assert _normal_quantile(0.5) == pytest.approx(0.0, abs=1e-9)
    assert _normal_quantile(0.975) == pytest.approx(1.959963985, abs=1e-6)
    assert _normal_quantile(0.025) == pytest.approx(-1.959963985, abs=1e-6)


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize(np.array([]))


def test_summarize_rejects_bad_confidence():
    with pytest.raises(ValueError):
        summarize(np.array([1.0, 2.0]), confidence=1.5)


def test_single_sample_has_zero_se():
    res = summarize(np.array([3.14]))
    assert res.std_error == 0.0
    assert res.confidence_interval == (3.14, 3.14)
