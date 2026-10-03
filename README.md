
## Description:

A stochastic retirement income model being built in distinct stages. The first stage is a simple deterministic baseline projection, after which I plan to add random market returns, lifespans (from mortality tables), correlation between inflation and market returns, and a comparison of withdrawal strategies.

## Purpose:

How likely is a retiree to run out of money, depending on their pension pot, equity returns, life expectancy and withdrawal strategy?

## Current state:

The current version simply projects a pension pot's value year by year depending on fixed assumptions: returns, withdrawals, inflation, fees, lifespan. It determines when/if the pot runs out, how many years of income it provides and how much is left at the end of a time horizon. It includes a sensitivity analysis of investment returns on the pension's performance. The main purpose of this stage is to provide a baseline which I can check later Monte Carlo simulations with.

![Deterministic projection of the pension pot](docs/images/projection.png)

![Sensitivity of the pension pot to the investment return](docs/images/return_sensitivity.png)

Notice how shifting the investment return by only 2% dramatically changes how long it takes to deplete the pot!

## Installation

In a terminal run:
```bash
git clone https://github.com/aroyston14/Retirement_income_Monte_Carlo_Simulator.git retirement-monte-carlo
cd retirement-monte-carlo
pip install -r requirements.txt
python run.py
python -m pytest
```

The projection plots will be produced in the 'outputs' folder.

The project is tested against closed-form results and a hand calculated projection.

## Assumptions and Limitations

The first stage of this project assumes fixed returns, inflation, withdrawals and lifespan, all of which will be modelled stochastically in later versions.

## Project Roadmap

- [x] Stage 1: Deterministic projection (fixed assumptions)
- [ ] Stage 2: Stochastic equity returns (GBM) and Monte Carlo simulation
- [ ] Stage 3: Random lifetimes from mortality tables
- [ ] Stage 4: Correlated bonds and inflation, calibrated to real data
- [ ] Stage 5: Comparison of withdrawal strategies, annuity pricing

