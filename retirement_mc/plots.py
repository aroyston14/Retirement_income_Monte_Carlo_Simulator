
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from retirement_mc.analytics import ruin_age, ruin_ages, prob_ruin, percentile_paths

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
    _save(fig, save_path)


# Saves the chart if a path is given, otherwise opens it in a window
def _save(fig, save_path):
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# Stage 2: Monte Carlo charts. pots has shape (n_sims, n_years + 1)

# Fan chart: percentile bands of the pot across all simulations, year by year
def plot_fan_chart(pots, start_age, deterministic_pot=None, save_path=None):
    ages = start_age + np.arange(pots.shape[1])
    p5, p25, p50, p75, p95 = percentile_paths(pots, (5, 25, 50, 75, 95))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(ages, p5, p95, color="tab:blue", alpha=0.15, label="5th–95th percentile")
    ax.fill_between(ages, p25, p75, color="tab:blue", alpha=0.35, label="25th–75th percentile")
    ax.plot(ages, p50, color="tab:blue", linewidth=2, label="Median")

    if deterministic_pot is not None:
        ax.plot(ages, deterministic_pot, color="black", linestyle="--", linewidth=1.5,
                label="Deterministic (fixed return)")

    p, low, high = prob_ruin(pots)
    title = (f"Monte Carlo projection: {len(pots):,} simulations, "
             f"P(ruin before {ages[-1]}) = {p:.1%}")
    _finish(fig, ax, ages, title, save_path)
    return fig, ax


# A sample of individual simulated paths, to make the randomness visible
def plot_sample_paths(pots, start_age, n_paths=50, save_path=None):
    ages = start_age + np.arange(pots.shape[1])

    fig, ax = plt.subplots(figsize=(10, 6))
    for pot in pots[:n_paths]:
        ax.plot(ages, pot, color="tab:blue", linewidth=0.8, alpha=0.3)
    ax.plot(ages, np.median(pots, axis=0), color="black", linewidth=2,
            label=f"Median of all {len(pots):,} simulations")

    _finish(fig, ax, ages, f"{n_paths} simulated paths of the pension pot", save_path)
    return fig, ax


# Histogram of the age at which the pot runs out, as a % of ALL simulations
def plot_ruin_age_histogram(pots, start_age, deterministic_pot=None, save_path=None):
    ages = ruin_ages(pots, start_age)
    ruined = ages[~np.isnan(ages)]
    end_age = start_age + pots.shape[1] - 1
    p, low, high = prob_ruin(pots)

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(start_age, end_age + 2) - 0.5          # one bar per whole age
    weights = np.full(len(ruined), 100 / len(pots))          # each path = its share of all sims
    ax.hist(ruined, bins=bins, weights=weights, color="tab:red", alpha=0.7,
            edgecolor="white", label="Simulations running out at this age")

    if deterministic_pot is not None:
        det_age = ruin_age(deterministic_pot, start_age)
        if det_age is not None:
            ax.axvline(det_age, color="black", linestyle="--", linewidth=1.5,
                       label=f"Deterministic ruin age ({det_age})")

    ax.set_xlabel("Age at which the pot runs out")
    ax.set_ylabel("% of all simulations")
    ax.set_title(f"When does the money run out? P(ruin before {end_age}) = {p:.1%} "
                 f"(95% CI {low:.1%}–{high:.1%})")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}%"))
    ax.set_xlim(start_age, end_age + 1)
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, save_path)
    return fig, ax
