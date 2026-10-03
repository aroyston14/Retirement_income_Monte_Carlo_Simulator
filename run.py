from retirement_mc.engine import project_pot
from retirement_mc.config import Assumptions
from retirement_mc.analytics import summary
from pathlib import Path
from dataclasses import replace
from retirement_mc.plots import plot_pot, plot_scenarios


def main():
    # Setting up our assumptions and running the projection
    a = Assumptions()
    pot = project_pot(a)

    # Displaying a summary of the projection results
    output = summary(pot, a)
    print(f'''
    Summary of retirement projection:
    =================================
    Starting pot: £{a.starting_pot:,.0f} at age {a.start_age}
    Annual investment returns: {a.annual_return:.1%}
    Annual withdrawal rate: {a.annual_withdrawal_rate:.1%} of starting pot, growing with inflation of {a.inflation:.1%}
    Annual fees: {a.annual_fee:.1%}
    Projection horizon: {a.start_age} to {a.end_age}
    ''')

    if output["ruin_age"] is not None:
        print(f'The pot ran out at age {output["ruin_age"]} after {output["years_of_withdrawals"]} years of withdrawals.')
    else:
        print(f'The pot lasted until the end of the horizon at age {a.end_age} with a final value of £{output["final_pot"]:,.0f}.')

    # Saving a plot of the projection
    Path("outputs").mkdir(exist_ok=True)
    plot_pot(pot, a.start_age, save_path="outputs/projection.png")

    # Sensitivity of the projection to the investment return assumption
    scenarios = {
        f"{r:.0%} return": project_pot(replace(a, annual_return=r))
        for r in (0.02, 0.04, 0.06, 0.08)
    }
    plot_scenarios(scenarios, a.start_age, title="Sensitivity to investment return",
                   save_path="outputs/return_sensitivity.png")

if __name__ == "__main__":
    main()
    
