"""The Monty Hall problem - conditional probability under asymmetric information.

Setup
-----
Three doors. One hides a car, two hide goats. The contestant picks a door
(call it A). The host - who knows where the car is - opens one of the
*other* two doors to reveal a goat (call the opened door B). The host
will never reveal the car.

The contestant is then offered the choice to *stay* with A or *switch*
to the remaining unopened door. Should you switch?

Most people say "it's 50/50, doesn't matter". They're wrong - and *why*
they're wrong is the lesson.

Why this is a Bayesian problem
------------------------------
The host's behavior is a *non-uniform sampling rule*: the host is
forbidden from opening the car door. That constraint leaks information
about where the car is. To compute P(car = remaining_unopened_door |
host opened B) you must condition on the host's policy, not just the
fact that a goat was revealed.

Why this matters for ML
-----------------------
Many ML bugs have the same shape: the data you see has been filtered by
some process you forgot to model, and conditioning on the post-filter
sample gives the wrong posterior. Selection bias, censoring, MAR vs
MNAR missingness, and Simpson's paradox all share this DNA.

Things to research
------------------
- Bayes' theorem with the host's "policy" as the likelihood.
- The phrase "Bertrand's box paradox" - same trick, different costume.
- "Monty Fall" variant: what if the host opens a door at *random* and
  it just happens to be a goat? The answer changes - find out why.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# TASK 1: derive the analytic answer
# ---------------------------------------------------------------------------
#
# Use Bayes' theorem with hypotheses H_A, H_B, H_C ("car is behind door
# A/B/C"), prior 1/3 each, and likelihood = P(host opens B | H_x).
# That likelihood depends on the host's policy:
#   * If car is at A (contestant's pick), host picks freely between B and C.
#   * If car is at B, host can't open B.
#   * If car is at C, host is forced to open B.
# Plug into Bayes; you'll find P(car = C | host opened B) is *not* 1/2.


def analytic_switch_probability() -> float:
    """Return P(win | switch) for the standard Monty Hall problem."""
    # TODO: Bayes' theorem on the three car-position hypotheses.
    raise NotImplementedError()


def analytic_stay_probability() -> float:
    """Return P(win | stay)."""
    # TODO: notice this one needs no Bayesian gymnastics - why?
    raise NotImplementedError()


# ---------------------------------------------------------------------------
# TASK 2: simulate the game and verify
# ---------------------------------------------------------------------------
#
# Each trial:
#   1. Place car uniformly at random behind one of three doors.
#   2. Contestant picks uniformly at random.
#   3. Host opens a door that is (a) not the contestant's pick AND
#      (b) not the car. If both remaining doors qualify (i.e. the
#      contestant guessed right), the host picks one uniformly.
#   4. Record whether stay/switch wins.
#
# Hint: the host's branch logic is where the conditional structure lives.
# Make sure your simulator implements the host's *policy*, not just "open
# a goat door at random and accept whatever".


def simulate(n_trials: int = 200_000, rng=None) -> dict[str, float]:
    """Return {'P(win | stay)': ..., 'P(win | switch)': ...}."""
    rng = rng or np.random.default_rng(0)
    # TODO: implement the simulation described above.
    raise NotImplementedError()


# ---------------------------------------------------------------------------


def main() -> None:
    print("=== Monty Hall ===")
    print(f"analytic  P(win | stay)   = {analytic_stay_probability():.5f}")
    print(f"analytic  P(win | switch) = {analytic_switch_probability():.5f}")
    sim = simulate()
    print(f"simulated P(win | stay)   = {sim['P(win | stay)']:.5f}")
    print(f"simulated P(win | switch) = {sim['P(win | switch)']:.5f}")


if __name__ == "__main__":
    main()
