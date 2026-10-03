# Testing the config assumptions

import dataclasses

import pytest

from retirement_mc.config import Assumptions


def test_defaults_construct():
    Assumptions()


def test_n_years():
    assert Assumptions(start_age=65, end_age=100).n_years == 35


def test_negative_pot_rejected():
    with pytest.raises(ValueError):
        Assumptions(starting_pot=-1.0)


def test_zero_pot_allowed():
    Assumptions(starting_pot=0.0)


@pytest.mark.parametrize("end_age", [65, 60])
def test_end_age_must_exceed_start_age(end_age):
    with pytest.raises(ValueError):
        Assumptions(start_age=65, end_age=end_age)


def test_assumptions_are_frozen():
    a = Assumptions()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.starting_pot = 1.0
