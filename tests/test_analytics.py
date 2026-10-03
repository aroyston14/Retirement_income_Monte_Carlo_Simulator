# Testing for the analytics

import numpy as np
import pytest

from retirement_mc.analytics import final_pot, ruin_age, summary, years_of_withdrawals
from retirement_mc.config import Assumptions
from retirement_mc.engine import project_pot


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
                    annual_return=0.04, annual_withdrawal_rate=0.06,
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
