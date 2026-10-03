# Testing the projection engine itself

import numpy as np
import pytest
from dataclasses import replace

from retirement_mc.config import Assumptions
from retirement_mc.engine import project_pot, simulate_pots
from retirement_mc.esg.equity import simulate_gross_returns


BASE = Assumptions(
    start_age=65,
    end_age=100,
    starting_pot=500_000.0,
    expected_annual_return=0.05,
    annual_withdrawal_rate=0.05,          
    inflation=0.02,
    annual_fee=0.005,
)


def make(**overrides) -> Assumptions:
    return replace(BASE, **overrides)


# testing shape and starting point

def test_output_length_is_n_years_plus_one():
    a = make()
    assert len(project_pot(a)) == a.n_years + 1


def test_first_value_is_starting_pot():
    assert project_pot(make())[0] == 500_000.0


# testing closed form special cases

def test_pure_compounding_matches_formula():
    # no withdrawals or fees
    a = make(annual_withdrawal_rate=0.0, annual_fee=0.0)
    t = np.arange(a.n_years + 1)
    expected = 500_000.0 * 1.05 ** t
    np.testing.assert_allclose(project_pot(a), expected)


def test_fee_only_matches_formula():
    # no returns, no withdrawals
    a = make(expected_annual_return=0.0, annual_withdrawal_rate=0.0)
    t = np.arange(a.n_years + 1)
    expected = 500_000.0 * 0.995 ** t
    np.testing.assert_allclose(project_pot(a), expected)


def test_linear_run_down_with_no_growth():
    # no returns, no fees, no inflation: pot runs down linearly to zero,

    a = make(start_age=65, end_age=80, starting_pot=100_000.0,
             annual_withdrawal_rate=0.10, expected_annual_return=0.0,
             inflation=0.0, annual_fee=0.0)
    pot = project_pot(a)
    expected = np.maximum(100_000.0 - 10_000.0 * np.arange(16), 0)
    np.testing.assert_allclose(pot, expected)
    assert pot[10] == 0


def test_withdrawals_grow_with_inflation():
    # no return or fee, 10% inflation
    a = make(start_age=65, end_age=68, starting_pot=10_000.0,
             annual_withdrawal_rate=0.10, expected_annual_return=0.0,
             inflation=0.10, annual_fee=0.0)
    np.testing.assert_allclose(project_pot(a), [10_000.0, 9_000.0, 7_900.0, 6_690.0])


# A handchecked example to make sure the engine is doing what we expect. The first two years are calculated by hand and then the rest of the projection is checked against a known good result.

def test_matches_hand_calculation_first_two_years():
    # Year 1: (500,000 - 25,000) * 1.05 * 0.995 = 496,256.25
    # Year 2: (496,256.25 - 25,500) * 1.05 * 0.995 = 491,822.5921875
    pot = project_pot(make())
    assert pot[1] == pytest.approx(496_256.25)
    assert pot[2] == pytest.approx(491_822.5921875)


# testing ruin behaviour

def test_pot_never_negative_with_huge_withdrawal():
    pot = project_pot(make(annual_withdrawal_rate=20.0))
    assert np.all(pot >= 0)
    assert pot[1] == 0


def test_once_ruined_pot_stays_at_zero():
    pot = project_pot(make(annual_withdrawal_rate=0.12))
    first_zero = np.argmax(pot == 0)
    assert pot[first_zero] == 0, "expected the pot to run out in this scenario"
    assert np.all(pot[first_zero:] == 0)


def test_higher_withdrawal_rate_never_leaves_more_money():
    low = project_pot(make(annual_withdrawal_rate=0.04))
    high = project_pot(make(annual_withdrawal_rate=0.06))
    assert np.all(high <= low)



# Stage 2: simulate_pots (Monte Carlo engine)


def constant_returns(a: Assumptions, n_sims: int) -> np.ndarray:
    # every simulation gets exactly 1 + r every year
    return np.full((n_sims, a.n_years), 1 + a.expected_annual_return)


# testing shape and starting point

def test_simulated_shape_comes_from_returns_array():
    # shape follows the returns passed in, not a.n_sims
    a = make(n_sims=10_000)
    returns = constant_returns(a, n_sims=3)
    assert simulate_pots(a, returns).shape == (3, a.n_years + 1)


def test_every_simulation_starts_at_starting_pot():
    a = make()
    pots = simulate_pots(a, constant_returns(a, n_sims=4))
    np.testing.assert_array_equal(pots[:, 0], 500_000.0)


# linking Stage 2 back to Stage 1

def test_constant_returns_reproduce_project_pot():
    a = make()
    pots = simulate_pots(a, constant_returns(a, n_sims=5))
    for row in pots:
        np.testing.assert_allclose(row, project_pot(a))


def test_zero_volatility_reproduces_project_pot():
    # the full pipeline (return generator + engine) with sigma = 0 must give
    # exactly the deterministic Stage 1 path in every simulation
    a = make(volatility=0.0, n_sims=50)
    returns = simulate_gross_returns(a, np.random.default_rng(a.seed))
    pots = simulate_pots(a, returns)
    np.testing.assert_allclose(pots, np.tile(project_pot(a), (50, 1)))


# hand-checked example

def test_matches_hand_calculation_with_hand_fed_returns():
    # pot 1,000, withdrawal 100 a year (10%), no inflation or fee, 2 years
    # sim 0: (1000 - 100) * 1.1 = 990,  then (990 - 100) * 0.9 = 801
    # sim 1: (1000 - 100) * 1.0 = 900,  then (900 - 100) * 1.0 = 800
    a = make(start_age=65, end_age=67, starting_pot=1_000.0,
             annual_withdrawal_rate=0.10, inflation=0.0, annual_fee=0.0)
    returns = np.array([[1.1, 0.9],
                        [1.0, 1.0]])
    expected = np.array([[1_000.0, 990.0, 801.0],
                         [1_000.0, 900.0, 800.0]])
    np.testing.assert_allclose(simulate_pots(a, returns), expected)


def test_hand_calculation_with_inflation_and_fee():
    # pot 1,000, withdrawal 100 growing 10%, fee 1%, returns 1.2 then 0.8
    # year 1: (1000 - 100) * 1.2 * 0.99 = 1069.2
    # year 2: (1069.2 - 110) * 0.8 * 0.99 = 759.6864
    a = make(start_age=65, end_age=67, starting_pot=1_000.0,
             annual_withdrawal_rate=0.10, inflation=0.10, annual_fee=0.01)
    pots = simulate_pots(a, np.array([[1.2, 0.8]]))
    np.testing.assert_allclose(pots[0], [1_000.0, 1_069.2, 759.6864])


# testing simulations behave independently

def test_each_row_depends_only_on_its_own_returns():
    # running one row on its own gives the same answer as running it in a batch
    a = make(volatility=0.15, n_sims=20)
    returns = simulate_gross_returns(a, np.random.default_rng(7))
    batch = simulate_pots(a, returns)
    single = simulate_pots(a, returns[[5]])
    np.testing.assert_allclose(single[0], batch[5])


def test_better_returns_never_leave_less_money():
    a = make(volatility=0.15, n_sims=200)
    returns = simulate_gross_returns(a, np.random.default_rng(3))
    worse = simulate_pots(a, returns)
    better = simulate_pots(a, returns * 1.01)
    assert np.all(better >= worse)


# testing ruin behaviour

def test_simulated_pots_never_negative():
    # high volatility and high withdrawals: lots of paths run out
    a = make(volatility=0.4, annual_withdrawal_rate=0.10, n_sims=1_000)
    pots = simulate_pots(a, simulate_gross_returns(a, np.random.default_rng(1)))
    assert np.all(pots >= 0)
    assert np.any(pots[:, -1] == 0), "expected some paths to run out in this scenario"


def test_simulated_pots_stay_at_zero_once_ruined():
    a = make(volatility=0.4, annual_withdrawal_rate=0.10, n_sims=1_000)
    pots = simulate_pots(a, simulate_gross_returns(a, np.random.default_rng(1)))
    hit_zero = pots == 0
    # once a column is zero for a row, every later column must also be zero
    ever_hit = np.logical_or.accumulate(hit_zero, axis=1)
    assert np.all(pots[ever_hit] == 0)
