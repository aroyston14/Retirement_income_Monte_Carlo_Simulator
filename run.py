from retirement_mc.engine import project_pot, simulate_pots
from retirement_mc.esg.equity import simulate_gross_returns
from retirement_mc.config import Assumptions
from retirement_mc.analytics import summary, mc_summary
from pathlib import Path
from dataclasses import replace
from retirement_mc.plots import (plot_pot, plot_scenarios, plot_fan_chart,
                                 plot_ruin_age_histogram, plot_sample_paths)
import numpy as np

def main():

    # Stage 1: deterministic projection:

    # Setting up our assumptions and running the projection
    a = Assumptions()
    pot = project_pot(a)

    # Displaying a summary of the projection results
    output = summary(pot, a)

    if output["ruin_age"] is not None:
        print(f'The pot ran out at age {output["ruin_age"]} after {output["years_of_withdrawals"]} years of withdrawals.')
    else:
        print(f'The pot lasted until the end of the horizon at age {a.end_age} with a final value of £{output["final_pot"]:,.0f}.')

    # Saving a plot of the projection
    Path("outputs").mkdir(exist_ok=True)
    plot_pot(pot, a.start_age, save_path="outputs/projection.png")

    # Sensitivity of the projection to the investment return assumption
    scenarios = {
        f"{r:.0%} return": project_pot(replace(a, expected_annual_return=r))
        for r in (0.02, 0.04, 0.06, 0.08)
    }
    plot_scenarios(scenarios, a.start_age, title="Sensitivity to investment return",
                   save_path="outputs/return_sensitivity.png")

    # Stage 2: Monte Carlo simulation of returns modelled as Geometric Brownian motion (GBM)
    rng = np.random.default_rng(a.seed)
    returns = simulate_gross_returns(a, rng)
    pots = simulate_pots(a, returns)
    s = mc_summary(pots, a)

    # Displaying a summary of the Monte Carlo results
    def fmt_age(age):
        return f"{age}" if age is not None else f"after {a.end_age}"

    low, high = s["prob_ruin_ci"]
    print(f'''
Monte Carlo simulation of returns (GBM)

Assumptions
  Starting pot:          £{a.starting_pot:,.0f} at age {a.start_age}
  First withdrawal:      £{a.annual_withdrawal_rate * a.starting_pot:,.0f} ({a.annual_withdrawal_rate:.1%} of starting pot), rising {a.inflation:.1%} p.a.
  Expected return:       {a.expected_annual_return:.1%} p.a., volatility {a.volatility:.1%}
  Annual fee:            {a.annual_fee:.1%}
  Simulations:           {s["n_sims"]:,} (seed {a.seed})

Results (projection to age {a.end_age})
  Probability of ruin:   {s["prob_ruin"]:.1%}  (95% CI {low:.1%} to {high:.1%})
  Median ruin age:       {fmt_age(s["median_ruin_age"])}  (deterministic: {fmt_age(output["ruin_age"])})
  Worst 5% run out by:   {fmt_age(s["ruin_age_5th_percentile"])}
  Final pot at {a.end_age}:      median £{s["median_final_pot"]:,.0f} | 5th pct £{s["final_pot_5th_percentile"]:,.0f} | 95th pct £{s["final_pot_95th_percentile"]:,.0f}
''')

    # Saving the Monte Carlo charts, with the deterministic path for comparison
    plot_fan_chart(pots, a.start_age, deterministic_pot=pot, save_path="outputs/mc_fan_chart.png")
    plot_ruin_age_histogram(pots, a.start_age, deterministic_pot=pot, save_path="outputs/mc_ruin_age_histogram.png")
    plot_sample_paths(pots, a.start_age, n_paths=50, save_path="outputs/mc_sample_paths.png")

if __name__ == "__main__":
    main()
    
