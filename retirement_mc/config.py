
# Creating our default assumptions for the initial deterministic model. These can be changed to assess different scenarios.

from dataclasses import dataclass
@dataclass(frozen = True)
class Assumptions:
    start_age: int = 65
    end_age: int = 100
    starting_pot: float = 140000.0
    annual_return : float = 0.04
    annual_withdrawal_rate: float = 0.06
    inflation: float = 0.02
    annual_fee: float = 0.005

    @property
    def n_years(self):
        return self.end_age - self.start_age

    def __post_init__(self):
        if self.starting_pot < 0:
            raise ValueError("Starting pot must be non-negative")
        elif self.end_age <= self.start_age:
            raise ValueError("End age must be greater than start age")
    
        
