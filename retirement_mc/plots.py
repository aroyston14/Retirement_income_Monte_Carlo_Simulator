"""
plots.py - Charts
=================

PURPOSE
    All the matplotlib code lives here, so the other modules stay free
    of plotting clutter.

AIM (Phase 1)
    - plot_pot(pot, start_age, save_path=None)
          A line chart of pot value (y) against age (x).
          Label the axes, format the y axis in GBP, add a title, and
          mark the ruin age with a vertical line if there is one.
          Save to outputs/ if save_path is given, otherwise show it.
    - Optional: plot several scenarios on one chart (e.g. 3%, 5% and 7%
      returns) to show sensitivity. This is a nice preview of Phase 2.

LATER PHASES
    Fan charts (percentile bands), histograms of the money left at
    death, and charts comparing strategies.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from retirement_mc.analytics import ruin_age 

# Plotting projection
def plot_pot(pot, start_age, save_path=None):
    
    ages = start_age + np.arange(len(pot))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ages, pot, linewidth=2, label="Pot value")

    age = ruin_age(pot, start_age)
    if age is not None:
        ax.axvline(age, color="red", linestyle="--", linewidth=1.5,
                   label=f"Pot runs out (age {age})")

    _finish(fig, ax, ages, "Projected pension pot (deterministic)", save_path)
    return fig, ax


# Plotting several projections on one chart to compare assumptions
def plot_scenarios(scenarios, start_age, title="Sensitivity of pension pot", save_path=None):
    """
    scenarios is a dict mapping a label to a pot array, e.g.
    {"3% return": pot_3, "4% return": pot_4}. Each pot is drawn as a line,
    with a dot on the x-axis where that pot runs out.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, pot in scenarios.items():
        ages = start_age + np.arange(len(pot))
        age = ruin_age(pot, start_age)
        label = f"{name} (runs out at {age})" if age is not None else f"{name} (lasts to {ages[-1]})"

        line, = ax.plot(ages, pot, linewidth=2, label=label)
        if age is not None:
            ax.plot(age, 0, "o", color=line.get_color(), markersize=8, clip_on=False)

    _finish(fig, ax, ages, title, save_path)
    return fig, ax


# Shared styling and saving, so every chart looks consistent
def _finish(fig, ax, ages, title, save_path):
    ax.set_xlabel("Age")
    ax.set_ylabel("Pot value")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value:,.0f}"))
    ax.set_xlim(ages[0], ages[-1])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()