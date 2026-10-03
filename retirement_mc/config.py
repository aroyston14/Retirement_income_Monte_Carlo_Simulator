
# Creating our default assumptions for the initial deterministic model. These can be changed to assess different scenarios.

from dataclasses import dataclass
@dataclass(frozen = True)
class Assumptions:
    start_age: int = 65
    end_age: int = 100
    starting_pot: float = 250000.0
    expected_annual_return : float = 0.05
    annual_withdrawal_rate: float = 0.04
    inflation: float = 0.02
    annual_fee: float = 0.005
    volatility: float = 0.10
    n_sims : int = 10000
    seed: int = 42

    @property
    def n_years(self):
        return self.end_age - self.start_age

    def __post_init__(self):
        if self.starting_pot < 0:
            raise ValueError("Starting pot must be non-negative")
        if self.volatility < 0:
            raise ValueError("Volatility must be non-negative")
        if self.n_sims <= 0:
            raise ValueError("Number of simulations must be positive")
        if self.annual_fee < 0 or self.annual_fee > 1:
            raise ValueError("Annual fee must be between 0 and 1")
        if self.annual_withdrawal_rate < 0:
            raise ValueError("Annual withdrawal rate must be non-negative")
        if self.end_age <= self.start_age:
            raise ValueError("End age must be greater than start age")
    
        
