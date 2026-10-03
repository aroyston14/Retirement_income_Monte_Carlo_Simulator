
# This deterministic projection engine takes in the assumptions and stores the size of the pension pot at the start of each year in an array. The first-year withdrawal is withdrawal_rate * starting_pot, and it then grows with inflation. Each year the withdrawal is subtracted from the previous year's pot and then the return and fee are applied. The final result is floored at zero.
import numpy as np
from retirement_mc.config import Assumptions

# Stage 1's deterministic projection engine.
def project_pot(a: Assumptions) -> np.ndarray:
    pot_array = np.zeros(a.n_years + 1)
    pot_array[0] = a.starting_pot
    initial_withdrawal = a.annual_withdrawal_rate * a.starting_pot
    for i in range(a.n_years):
        actual_value = (pot_array[i] - initial_withdrawal * (1 + a.inflation) ** i) * (1 + a.expected_annual_return) * (1 - a.annual_fee)
        pot_array[i + 1] = max(actual_value, 0)
    return pot_array

# Stage 2's Monte Carlo projection engine. This stochastically models the returns instead of treating them as a constant rate.
def simulate_pots(a: Assumptions, gross_returns: np.ndarray) -> np.ndarray:
    n_sims, n_years = gross_returns.shape
    pots = np.zeros((n_sims, n_years + 1))
    pots[:, 0] = a.starting_pot
    initial_withdrawal = a.annual_withdrawal_rate * a.starting_pot
    for t in range(n_years):
        withdrawal = initial_withdrawal * (1 + a.inflation) ** t
        pots[:, t + 1] = np.maximum((pots[:, t] - withdrawal) * gross_returns[:, t] * (1 - a.annual_fee), 0)
    return pots
    
