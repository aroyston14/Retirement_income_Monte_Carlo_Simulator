
# Interpreting and summarising the results of a projection

import numpy as np


# Stage 1: a single deterministic path

def final_pot(pot) -> float:
    return pot[-1]

def years_of_withdrawals(pot) -> int:
    return sum(1 for p in pot[:-1] if p > 0)  # last value is the end of the horizon: no withdrawal from it

def ruin_age(pot, start_age) -> int | None:
    for t, value in enumerate(pot):
        if value <= 0:
            return start_age + t
    return None 

def summary(pot, a):
      return {
            "final_pot": final_pot(pot),
            "years_of_withdrawals": years_of_withdrawals(pot),
            "ruin_age": ruin_age(pot, a.start_age),
      }

# Stage 2: many simulated paths, shape (n_sims, n_years + 1)

def ruin_ages(pots, start_age) -> np.ndarray:
    """Age at which each path first hits zero, or nan if it never does."""
    hit = pots <= 0
    first = hit.argmax(axis=1)
    return np.where(hit.any(axis=1), start_age + first, np.nan)


def prob_ruin(pots, z=1.96):
    """Estimated probability of ruin with a confidence interval (default 95%)."""
    ruined = (pots <= 0).any(axis=1)
    n = len(ruined)
    p = float(ruined.mean())
    se = float(np.sqrt(p * (1 - p) / n))
    return p, max(p - z * se, 0.0), min(p + z * se, 1.0)


def percentile_paths(pots, percentiles=(5, 25, 50, 75, 95)) -> np.ndarray:
    """One row per percentile, one column per year: the bands of a fan chart."""
    return np.percentile(pots, percentiles, axis=0)


def ruin_age_percentile(ages, q):
    """
    The age by which q% of ALL paths have run out, or None if fewer than q% do
    before the horizon. Paths that never run out count as running out 'after
    the horizon' (infinity), rather than being dropped.
    """
    value = np.percentile(np.nan_to_num(ages, nan=np.inf), q, method="inverted_cdf")
    return None if np.isinf(value) else int(value)


def mc_summary(pots, a) -> dict:
    p, low, high = prob_ruin(pots)
    ages = ruin_ages(pots, a.start_age)
    final = pots[:, -1]
    return {
        "n_sims": len(pots),
        "prob_ruin": p,
        "prob_ruin_ci": (low, high),
        "median_ruin_age": ruin_age_percentile(ages, 50),
        "ruin_age_5th_percentile": ruin_age_percentile(ages, 5),
        "median_final_pot": float(np.median(final)),
        "final_pot_5th_percentile": float(np.percentile(final, 5)),
        "final_pot_95th_percentile": float(np.percentile(final, 95)),
    }
