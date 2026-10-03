import numpy as np
from retirement_mc.config import Assumptions


def simulate_gross_returns(a: Assumptions, rng: np.random.Generator) -> np.ndarray:
    Z = rng.standard_normal((a.n_sims, a.n_years))
    mu = np.log(1 + a.expected_annual_return)
    sigma = a.volatility
    return np.exp((mu - 0.5 * sigma**2) + sigma * Z)