# Testing the random equity return generator (GBM)

from dataclasses import replace

import numpy as np
import pytest

from retirement_mc.config import Assumptions
from retirement_mc.esg.equity import simulate_gross_returns


BASE = Assumptions(
    start_age=65,
    end_age=100,
    expected_annual_return=0.04,
    volatility=0.12,
    n_sims=1_000,
    seed=42,
)


def make(**overrides) -> Assumptions:
    return replace(BASE, **overrides)


def returns_for(a: Assumptions) -> np.ndarray:
    return simulate_gross_returns(a, np.random.default_rng(a.seed))


# testing shape and basic properties

def test_shape_is_n_sims_by_n_years():
    a = make(n_sims=500)
    assert returns_for(a).shape == (500, a.n_years)


def test_returns_always_positive():
    # lognormal returns can never be zero or negative, even with high volatility
    assert np.all(returns_for(make(volatility=0.5)) > 0)


def test_zero_volatility_gives_constant_return():
    # sigma = 0 removes all randomness: every year returns exactly 1 + r,
    # which is what lets the Monte Carlo model reproduce Stage 1
    R = returns_for(make(volatility=0.0))
    np.testing.assert_allclose(R, 1.04)


# testing reproducibility

def test_same_seed_gives_identical_returns():
    a = make()
    R1 = simulate_gross_returns(a, np.random.default_rng(123))
    R2 = simulate_gross_returns(a, np.random.default_rng(123))
    np.testing.assert_array_equal(R1, R2)


def test_different_seeds_give_different_returns():
    a = make()
    R1 = simulate_gross_returns(a, np.random.default_rng(1))
    R2 = simulate_gross_returns(a, np.random.default_rng(2))
    assert not np.array_equal(R1, R2)


def test_repeated_calls_on_one_generator_give_fresh_draws():
    # the generator moves on after each call, so later draws are new numbers
    a = make()
    rng = np.random.default_rng(42)
    assert not np.array_equal(simulate_gross_returns(a, rng), simulate_gross_returns(a, rng))


# testing statistical properties (20,000 sims x 35 years = 700,000 samples)

LARGE = make(n_sims=20_000)


def test_mean_return_matches_expected_return():
    # E[R] = 1 + expected_annual_return: the -sigma^2/2 correction is what makes this hold
    assert returns_for(LARGE).mean() == pytest.approx(1.04, rel=1e-3)


def test_log_returns_have_correct_mean_and_volatility():
    # ln R ~ N(mu - sigma^2/2, sigma^2), with mu = ln(1.04)
    log_R = np.log(returns_for(LARGE))
    assert log_R.mean() == pytest.approx(np.log(1.04) - 0.5 * 0.12**2, abs=1e-3)
    assert log_R.std() == pytest.approx(0.12, rel=1e-2)


def test_median_below_mean_volatility_drag():
    # median of a lognormal is exp(mu - sigma^2/2), below the mean exp(mu)
    R = returns_for(LARGE)
    expected_median = np.exp(np.log(1.04) - 0.5 * 0.12**2)
    assert np.median(R) == pytest.approx(expected_median, rel=1e-3)
    assert np.median(R) < R.mean()


def test_higher_volatility_widens_spread_not_mean():
    low = returns_for(make(n_sims=20_000, volatility=0.05))
    high = returns_for(make(n_sims=20_000, volatility=0.20))
    assert high.std() > low.std()
    assert high.mean() == pytest.approx(low.mean(), rel=2e-3)
