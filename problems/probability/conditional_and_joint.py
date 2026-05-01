"""Conditional, joint, and marginal probability - three problems to work.

Definitions to keep in mind
---------------------------
* Joint:        P(A, B) = the probability A and B both occur.
* Marginal:     P(A) = sum_b P(A, B=b)        ("law of total probability")
                     = sum_b P(A | B=b) * P(B=b).
* Conditional:  P(A | B) = P(A, B) / P(B), assuming P(B) > 0.
* Chain rule:   P(A, B) = P(A | B) * P(B) = P(B | A) * P(A).
* Independence: A _||_ B iff P(A, B) = P(A) * P(B), equivalently
                P(A | B) = P(A).

For each problem below, fill in two functions:

    1. analytic_*(...)   -> compute the closed-form answer with arithmetic
                            on the rules above (no random numbers).
    2. simulate_*(...)   -> draw many samples and return empirical
                            frequencies. With enough trials, this should
                            agree with your analytic answer to ~0.5%.

If your analytic and simulated numbers disagree, one of the two has a
bug - and which one is the bug is *itself* informative. Diagnose before
moving on.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Problem 1: joint distribution of two fair coin flips
# ---------------------------------------------------------------------------
#
# Two fair coins are flipped independently. Let
#   A = "first coin is heads",
#   B = "second coin is heads".
#
# TASKS
# -----
# Compute P(A, B), P(A), and P(A | B). One of these three reduces to a
# trivial number because of independence - which one, and why?
#
# Hint: write down the 2x2 joint table by hand first, then derive the
# marginal and the conditional from it.


def analytic_two_coin() -> dict[str, float]:
    """Return {'P(A, B)': ..., 'P(A)': ..., 'P(A | B)': ...}."""
    # TODO: fill in the three numbers using the definitions above.
    raise NotImplementedError()


def simulate_two_coin(n_trials: int = 200_000, rng=None) -> dict[str, float]:
    """Return the same three quantities estimated from random flips.

    Hint: draw a (n_trials, 2) array of bits, then average the relevant
    boolean masks. P(A | B) is the mean of A *restricted to rows where
    B is true* - this is the "conditioning = filtering" intuition.
    """
    rng = rng or np.random.default_rng(0)
    # TODO: implement.
    raise NotImplementedError()


# ---------------------------------------------------------------------------
# Problem 2: drawing two cards without replacement
# ---------------------------------------------------------------------------
#
# Draw two cards from a 52-card deck *without replacement*. Let
#   A = "first card is a heart",
#   B = "second card is a heart".
#
# TASKS
# -----
# Compute P(A), P(B | A), and P(A, B). Where in your derivation does the
# fact that A and B are *not* independent show up?
#
# Note: P(A | B) is not the same shape as P(B | A) in general - to flip
# the conditioning you need Bayes' theorem (Problem stack: bayes_theorem.py).


def analytic_two_card() -> dict[str, float]:
    """Return {'P(A)': ..., 'P(B | A)': ..., 'P(A, B)': ...}."""
    # TODO: apply the chain rule P(A, B) = P(A) * P(B | A). Count cards.
    raise NotImplementedError()


def simulate_two_card(n_trials: int = 200_000, rng=None) -> dict[str, float]:
    """Estimate the three quantities by sampling 2-card hands.

    Hint: encode the deck as 13 ones (hearts) + 39 zeros (everything else)
    and use rng.choice(deck, size=2, replace=False). Track A_count and
    A_and_B_count as you go.
    """
    rng = rng or np.random.default_rng(1)
    # TODO: implement.
    raise NotImplementedError()


# ---------------------------------------------------------------------------
# Problem 3: medical test - marginalization in action
# ---------------------------------------------------------------------------
#
# A disease has prevalence P(D) = 0.01.
# A test has sensitivity P(T | D) = 0.99
# and false-positive rate P(T | ~D) = 0.05.
#
# TASKS
# -----
# Compute the *marginal* probability of testing positive, P(T), via the
# law of total probability. (Resist the urge to compute P(D | T) here -
# that's the next exercise.)
#
# Hint: P(T) = P(T | D) * P(D) + P(T | ~D) * P(~D). Plug in numbers.


def analytic_test_marginal() -> float:
    """Return P(T)."""
    # TODO: implement using the law of total probability.
    raise NotImplementedError()


def simulate_test_marginal(n_trials: int = 500_000, rng=None) -> float:
    """Sample (D, T) jointly and return the empirical P(T).

    Hint: use np.where to make T conditional on D - this mirrors the
    generative process the analytic formula sums over.
    """
    rng = rng or np.random.default_rng(2)
    # TODO: implement.
    raise NotImplementedError()


# ---------------------------------------------------------------------------
# Driver - runs your solutions and prints them side by side.
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== Problem 1 - two fair coins ===")
    print("analytic ", analytic_two_coin())
    print("simulated", simulate_two_coin())

    print("\n=== Problem 2 - two cards w/o replacement ===")
    print("analytic ", analytic_two_card())
    print("simulated", simulate_two_card())

    print("\n=== Problem 3 - medical test marginal ===")
    print(f"analytic  P(T) = {analytic_test_marginal():.5f}")
    print(f"simulated P(T) = {simulate_test_marginal():.5f}")


if __name__ == "__main__":
    main()
