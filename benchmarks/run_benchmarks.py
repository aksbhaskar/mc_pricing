"""Benchmarks: Monte Carlo convergence and variance-reduction efficiency.

Running this script (``python benchmarks/run_benchmarks.py``) produces:

1. ``convergence.png``: log-log plot of absolute pricing error against the
   number of paths for plain MC, antithetic, control variate, and the combined
   estimator, overlaid with the theoretical :math:`O(N^{-1/2})` reference line.
2. ``variance_reduction.png``: the variance-reduction ratio achieved by each
   technique.
3. A console table of prices, standard errors, and wall-clock timings.

The plots are what the README embeds; the console table is a quick sanity
check that the estimators agree with Black-Scholes.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from mc_pricing import OptionSpec, OptionType, black_scholes as bs, monte_carlo_price

HERE = Path(__file__).resolve().parent

# A representative at-the-money option with a dividend yield.
SPEC = OptionSpec(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    volatility=0.20,
    option_type=OptionType.CALL,
    dividend_yield=0.02,
)

METHODS = {
    "Plain MC": dict(antithetic=False, control_variate=False),
    "Antithetic": dict(antithetic=True, control_variate=False),
    "Control variate": dict(antithetic=False, control_variate=True),
    "Antithetic + CV": dict(antithetic=True, control_variate=True),
}

PATH_COUNTS = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]


def run_convergence(n_trials: int = 40) -> dict[str, list[float]]:
    """Root-mean-square pricing error vs. path count, averaged over trials.

    Averaging the error over independent seeds smooths the curves so the
    :math:`O(N^{-1/2})` slope is visible rather than buried in single-run noise.
    """
    analytic = bs.price(SPEC)
    rms_error = {name: [] for name in METHODS}

    for n_paths in PATH_COUNTS:
        for name, kwargs in METHODS.items():
            errors = []
            for trial in range(n_trials):
                res = monte_carlo_price(
                    SPEC, n_paths=n_paths, seed=1000 * trial + n_paths, **kwargs
                )
                errors.append((res.price - analytic) ** 2)
            rms_error[name].append(float(np.sqrt(np.mean(errors))))
        print(f"  paths={n_paths:>7,}  done")
    return rms_error


def print_summary_table() -> None:
    """Print a price/SE/timing table at a fixed large path count."""
    analytic = bs.price(SPEC)
    n_paths = 500_000
    print(f"\nAnalytic Black-Scholes price: {analytic:.6f}")
    print(f"\n{'Method':<20}{'Price':>12}{'Abs err':>12}{'Std err':>12}"
          f"{'VRR':>8}{'Time (ms)':>12}")
    print("-" * 76)
    for name, kwargs in METHODS.items():
        t0 = time.perf_counter()
        res = monte_carlo_price(SPEC, n_paths=n_paths, seed=0, **kwargs)
        dt = (time.perf_counter() - t0) * 1e3
        print(f"{name:<20}{res.price:>12.5f}{abs(res.price - analytic):>12.2e}"
              f"{res.std_error:>12.2e}{res.variance_reduction_ratio:>8.1f}{dt:>12.1f}")


def make_plots(rms_error: dict[str, list[float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = np.array(PATH_COUNTS, dtype=float)
    colors = {
        "Plain MC": "#4C72B0",
        "Antithetic": "#DD8452",
        "Control variate": "#55A868",
        "Antithetic + CV": "#C44E52",
    }

    # --- Convergence plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, errs in rms_error.items():
        ax.loglog(paths, errs, "o-", color=colors[name], label=name, markersize=4)
    # Theoretical 1/sqrt(N) reference anchored to plain MC's first point.
    ref = rms_error["Plain MC"][0] * np.sqrt(paths[0] / paths)
    ax.loglog(paths, ref, "k--", alpha=0.5, label=r"$O(N^{-1/2})$ reference")
    ax.set_xlabel("Number of paths $N$")
    ax.set_ylabel("RMS absolute pricing error")
    ax.set_title("Monte Carlo convergence to Black-Scholes (ATM call)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "convergence.png", dpi=130)
    print(f"\nSaved {HERE / 'convergence.png'}")

    # --- Variance-reduction ratio plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    baseline = np.array(rms_error["Plain MC"]) ** 2
    for name, errs in rms_error.items():
        if name == "Plain MC":
            continue
        ratio = baseline / (np.array(errs) ** 2)
        ax.semilogx(paths, ratio, "o-", color=colors[name], label=name, markersize=4)
    ax.axhline(1.0, color="k", ls="--", alpha=0.5, label="No reduction")
    ax.set_xlabel("Number of paths $N$")
    ax.set_ylabel("Variance-reduction ratio (vs. plain MC)")
    ax.set_title("Efficiency gain from variance reduction (ATM call)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "variance_reduction.png", dpi=130)
    print(f"Saved {HERE / 'variance_reduction.png'}")


def main() -> None:
    print("Running convergence study...")
    rms_error = run_convergence()
    print_summary_table()
    try:
        make_plots(rms_error)
    except ImportError:
        print("\nmatplotlib not installed; skipping plots "
              "(install with `pip install matplotlib`).")


if __name__ == "__main__":
    main()
