"""End-to-end demonstration of the mc_pricing library.

Run with ``python examples/demo.py``. It prices a vanilla European option four
ways (analytic + three Monte Carlo variance-reduction configurations), reports
Greeks from both closed-form and Monte Carlo estimators side by side, and
recovers an implied volatility.
"""

from __future__ import annotations

from mc_pricing import (
    OptionSpec,
    OptionType,
    black_scholes as bs,
    implied_volatility,
    monte_carlo_greeks,
    monte_carlo_price,
)


def section(title: str) -> None:
    print("\n" + title)
    print("=" * len(title))


def main() -> None:
    spec = OptionSpec(
        spot=100.0,
        strike=105.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        option_type=OptionType.CALL,
        dividend_yield=0.02,
    )

    section("Option")
    print(f"  {spec.option_type.value.upper()}  S0={spec.spot}  K={spec.strike}  "
          f"T={spec.maturity}y  r={spec.rate:.0%}  q={spec.dividend_yield:.0%}  "
          f"sigma={spec.volatility:.0%}")

    # --- Pricing ---
    section("Pricing")
    analytic = bs.price(spec)
    print(f"  Analytic Black-Scholes : {analytic:.6f}")

    configs = [
        ("Plain Monte Carlo", dict()),
        ("Antithetic variates", dict(antithetic=True)),
        ("Control variate", dict(control_variate=True)),
        ("Antithetic + control", dict(antithetic=True, control_variate=True)),
    ]
    for label, kwargs in configs:
        res = monte_carlo_price(spec, n_paths=500_000, seed=0, **kwargs)
        lo, hi = res.confidence_interval
        print(f"  {label:<22}: {res.price:.6f}  "
              f"95% CI=[{lo:.4f}, {hi:.4f}]  "
              f"SE={res.std_error:.2e}  "
              f"variance-reduction x{res.variance_reduction_ratio:.1f}")

    # --- Greeks ---
    section("Greeks (closed-form vs. Monte Carlo)")
    cf = bs.greeks(spec)
    mc = monte_carlo_greeks(spec, n_paths=500_000, seed=1)
    print(f"  {'Greek':<8}{'Black-Scholes':>16}{'Monte Carlo':>16}"
          f"{'MC std err':>14}   method")
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        print(f"  {name:<8}{cf[name]:>16.5f}{mc[name].value:>16.5f}"
              f"{mc[name].std_error:>14.2e}   {mc[name].method}")

    # --- Implied volatility ---
    section("Implied volatility")
    market_price = analytic  # pretend the analytic price is a market quote
    iv = implied_volatility(
        market_price, spec.spot, spec.strike, spec.maturity, spec.rate,
        spec.option_type, dividend_yield=spec.dividend_yield,
    )
    print(f"  Quoted price {market_price:.6f}  ->  implied vol {iv:.4%} "
          f"(true {spec.volatility:.4%})")


if __name__ == "__main__":
    main()
