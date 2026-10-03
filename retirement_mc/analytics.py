
# Interpreting and summarising the results of a projection

def final_pot(pot) -> float:
    return pot[-1]

def years_of_withdrawals(pot) -> int:
    return sum(1 for p in pot[:-1] if p > 0)  # last value is the end of the horizon: no withdrawal from it

def ruin_age(pot, start_age) -> int | None:
    for t, value in enumerate(pot):
        if value <= 0:
            return start_age + t
    return None 

def summary(pot, a):
      return {
            "final_pot": final_pot(pot),
            "years_of_withdrawals": years_of_withdrawals(pot),
            "ruin_age": ruin_age(pot, a.start_age),
      }