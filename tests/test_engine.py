# Testing the projection engine itself

import numpy as np
import pytest
from dataclasses import replace

from retirement_mc.config import Assumptions
from retirement_mc.engine import project_pot


BASE = Assumptions(
    start_age=65,
    end_age=100,
    starting_pot=500_000.0,
    annual_return=0.05,
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
    a = make(annual_return=0.0, annual_withdrawal_rate=0.0)
    t = np.arange(a.n_years + 1)
    expected = 500_000.0 * 0.995 ** t
    np.testing.assert_allclose(project_pot(a), expected)


def test_linear_run_down_with_no_growth():
    # no returns, no fees, no inflation: pot runs down linearly to zero,

    a = make(start_age=65, end_age=80, starting_pot=100_000.0,
             annual_withdrawal_rate=0.10, annual_return=0.0,
             inflation=0.0, annual_fee=0.0)
    pot = project_pot(a)
    expected = np.maximum(100_000.0 - 10_000.0 * np.arange(16), 0)
    np.testing.assert_allclose(pot, expected)
    assert pot[10] == 0


def test_withdrawals_grow_with_inflation():
    # no return or fee, 10% inflation
    a = make(start_age=65, end_age=68, starting_pot=10_000.0,
             annual_withdrawal_rate=0.10, annual_return=0.0,
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
