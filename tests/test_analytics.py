# Testing the analytics

import numpy as np
import pytest

from dataclasses import replace

from retirement_mc.analytics import (final_pot, ruin_age, summary, years_of_withdrawals,
                                     ruin_ages, prob_ruin, percentile_paths,
                                     ruin_age_percentile, mc_summary)
from retirement_mc.config import Assumptions
from retirement_mc.engine import project_pot, simulate_pots
from retirement_mc.esg.equity import simulate_gross_returns


RUINED = np.array([100.0, 60.0, 20.0, 0.0, 0.0])      
SURVIVES = np.array([100.0, 90.0, 80.0, 70.0, 60.0])  
EMPTY = np.array([0.0, 0.0, 0.0])                     


# ruin age tests

def test_ruin_age_first_zero():
    assert ruin_age(RUINED, start_age=65) == 68


def test_ruin_age_none_if_never_runs_out():
    assert ruin_age(SURVIVES, start_age=65) is None


def test_ruin_age_is_start_age_if_pot_starts_empty():
    assert ruin_age(EMPTY, start_age=65) == 65


# years of withdrawals tests

def test_years_of_withdrawals_when_ruined():
    # Withdrawals taken at indices 0, 1, 2 (the last one partial)
    assert years_of_withdrawals(RUINED) == 3


def test_years_of_withdrawals_when_pot_survives():
    # 5 values = 4 years, so 4 withdrawals (not 5)
    assert years_of_withdrawals(SURVIVES) == 4


def test_years_of_withdrawals_when_pot_starts_empty():
    assert years_of_withdrawals(EMPTY) == 0


def test_years_of_withdrawals_never_exceeds_projection_years():
    a = Assumptions(annual_withdrawal_rate=0.0)
    assert years_of_withdrawals(project_pot(a)) == a.n_years


# final pot tests

def test_final_pot():
    assert final_pot(SURVIVES) == 60.0
    assert final_pot(RUINED) == 0.0


# testing the summary function

def test_summary_for_default_assumptions():
    # £140k, 6% withdrawal, 4% return, 2% inflation, 0.5% fee:
    # the pot first hits zero at index 19, i.e. age 84.
    a = Assumptions(start_age=65, end_age=100, starting_pot=140_000.0,
                    expected_annual_return=0.04, annual_withdrawal_rate=0.06,
                    inflation=0.02, annual_fee=0.005)
    s = summary(project_pot(a), a)
    assert s["ruin_age"] == 84
    assert s["years_of_withdrawals"] == 19
    assert s["final_pot"] == pytest.approx(0.0)


def test_summary_consistency_when_pot_survives():
    a = Assumptions(annual_withdrawal_rate=0.03)
    s = summary(project_pot(a), a)
    assert s["ruin_age"] is None
    assert s["years_of_withdrawals"] == a.n_years
    assert s["final_pot"] > 0

# Stage 2: Monte Carlo analytics on many paths, shape (n_sims, n_years + 1)

# Used an LLM to generate these handwritten tests before I checked them to save time. They are not guaranteed to be correct, but they are a good sanity check for the analytics functions.

# 4 hand-made simulations over 3 years, starting at age 65
POTS = np.array([
    [100.0, 50.0,  0.0,  0.0],   # runs out at index 2 -> age 67
    [100.0, 80.0, 60.0, 40.0],   # never runs out
    [100.0,  0.0,  0.0,  0.0],   # runs out at index 1 -> age 66
    [100.0, 90.0, 80.0, 70.0],   # never runs out
])


# ruin_ages

def test_ruin_ages_per_simulation():
    np.testing.assert_array_equal(ruin_ages(POTS, start_age=65), [67, np.nan, 66, np.nan])


def test_ruin_ages_all_survive_gives_all_nan():
    assert np.all(np.isnan(ruin_ages(POTS[[1, 3]], start_age=65)))


def test_ruin_ages_matches_stage1_ruin_age_row_by_row():
    for row, age in zip(POTS, ruin_ages(POTS, start_age=65)):
        expected = ruin_age(row, 65)
        assert (np.isnan(age) and expected is None) or age == expected


# prob_ruin

def test_prob_ruin_and_confidence_interval():
    # p = 2/4 = 0.5, se = sqrt(0.5 * 0.5 / 4) = 0.25, CI = 0.5 +/- 1.96 * 0.25
    p, low, high = prob_ruin(POTS)
    assert p == 0.5
    assert low == pytest.approx(0.01)
    assert high == pytest.approx(0.99)


def test_prob_ruin_zero_and_one():
    assert prob_ruin(POTS[[1, 3]]) == (0.0, 0.0, 0.0)
    assert prob_ruin(POTS[[0, 2]]) == (1.0, 1.0, 1.0)


def test_confidence_interval_clipped_to_0_1():
    # 1 of 2 ruined: 0.5 +/- 0.69 would go outside [0, 1]
    assert prob_ruin(POTS[[0, 1]]) == (0.5, 0.0, 1.0)


def test_confidence_interval_contains_p_and_narrows_with_more_sims():
    p_small, low_small, high_small = prob_ruin(POTS)
    p_big, low_big, high_big = prob_ruin(np.tile(POTS, (100, 1)))   # same p, 100x the sims
    assert p_small == p_big
    assert low_big <= p_big <= high_big
    # 100x the simulations -> interval 10x narrower (1 / sqrt(n))
    assert (high_big - low_big) == pytest.approx((high_small - low_small) / 10)


def test_z_controls_interval_width():
    assert prob_ruin(POTS, z=0) == (0.5, 0.5, 0.5)


# percentile_paths

def test_percentile_paths_shape():
    assert percentile_paths(POTS).shape == (5, 4)


def test_percentile_paths_values():
    # one column holding 0, 1, ..., 100: the q-th percentile is exactly q
    pots = np.tile(np.arange(101.0)[:, None], (1, 3))
    bands = percentile_paths(pots, percentiles=(5, 50, 95))
    np.testing.assert_allclose(bands, [[5, 5, 5], [50, 50, 50], [95, 95, 95]])


def test_percentile_bands_are_ordered():
    a = Assumptions(n_sims=2_000)
    pots = simulate_pots(a, simulate_gross_returns(a, np.random.default_rng(0)))
    bands = percentile_paths(pots)
    assert np.all(np.diff(bands, axis=0) >= 0)   # 5th <= 25th <= ... <= 95th every year


# ruin_age_percentile

def test_ruin_age_percentile_counts_survivors():
    # sorted with survivors as infinity: [66, 67, inf, inf]
    ages = ruin_ages(POTS, start_age=65)
    assert ruin_age_percentile(ages, 25) == 66
    assert ruin_age_percentile(ages, 50) == 67
    assert ruin_age_percentile(ages, 75) is None   # 75% of paths haven't run out


def test_ruin_age_percentile_does_not_drop_survivors():
    # only 1 of 4 runs out: the median retiree never runs out (dropping
    # survivors would wrongly report a median ruin age of 70)
    ages = np.array([70.0, np.nan, np.nan, np.nan])
    assert ruin_age_percentile(ages, 50) is None


def test_ruin_age_percentile_all_survive():
    assert ruin_age_percentile(np.array([np.nan, np.nan]), 5) is None


# mc_summary

def test_mc_summary_on_hand_made_paths():
    a = Assumptions(start_age=65, end_age=68)
    s = mc_summary(POTS, a)
    assert s["n_sims"] == 4
    assert s["prob_ruin"] == 0.5
    assert s["prob_ruin_ci"] == pytest.approx((0.01, 0.99))
    assert s["median_ruin_age"] == 67
    assert s["ruin_age_5th_percentile"] == 66
    # final pots [0, 40, 0, 70]
    assert s["median_final_pot"] == 20.0
    assert s["final_pot_5th_percentile"] == 0.0
    assert s["final_pot_95th_percentile"] == pytest.approx(65.5)


def test_mc_summary_with_zero_volatility_matches_stage1():
    # sigma = 0: every path is the deterministic one, which runs out at 84
    a = Assumptions(start_age=65, end_age=100, starting_pot=140_000.0,
                    expected_annual_return=0.04, annual_withdrawal_rate=0.06,
                    inflation=0.02, annual_fee=0.005, volatility=0.0, n_sims=100)
    pots = simulate_pots(a, simulate_gross_returns(a, np.random.default_rng(a.seed)))
    s = mc_summary(pots, a)
    stage1_age = ruin_age(project_pot(a), a.start_age)
    assert s["prob_ruin"] == 1.0
    assert s["median_ruin_age"] == stage1_age
    assert s["ruin_age_5th_percentile"] == stage1_age


def test_lower_withdrawal_rate_lowers_prob_ruin():
    # same random returns, smaller withdrawals -> fewer paths run out
    high = Assumptions(annual_withdrawal_rate=0.06, n_sims=2_000)
    low = replace(high, annual_withdrawal_rate=0.04)
    returns = simulate_gross_returns(high, np.random.default_rng(5))
    assert prob_ruin(simulate_pots(low, returns))[0] < prob_ruin(simulate_pots(high, returns))[0]
