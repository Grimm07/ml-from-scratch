"""Bayes' theorem - inverting conditional probabilities.

Bayes' theorem
--------------
    P(H | E) = P(E | H) * P(H) / P(E)

where
    P(H)     prior - belief in H before seeing the evidence
    P(E | H) likelihood - how plausible E is if H is true
    P(E)     marginal likelihood / "evidence" - the normalizing constant
    P(H | E) posterior - updated belief in H after seeing E.

Why this matters for ML
-----------------------
Naive Bayes, Bayesian linear/logistic regression, Bayesian networks,
MAP estimation, EM, and variational inference are all "compute the
posterior" problems. The disease-test example below is the cleanest way
to internalize *why* posterior probabilities can be very different from
the conditional probabilities you started with.

Things to research while solving
--------------------------------
- "Base rate fallacy" - why ignoring the prior gives wildly wrong answers.
- The denominator P(E) usually comes from the law of total probability;
  see problems/probability/conditional_and_joint.py problem 3 for that.
- For Problem 2, look up "sequential Bayesian updating" - the posterior
  after observation k becomes the prior for observation k+1.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Problem 1: rare-disease test (the classic counter-intuitive Bayes example)
# ---------------------------------------------------------------------------
#
# Same setup as conditional_and_joint.py problem 3:
#   P(D) = 0.01,   P(T | D) = 0.99,   P(T | ~D) = 0.05.
#
# TASK
# ----
# Given a positive test, what is the probability of disease, P(D | T)?
# Most people guess >= 0.9 - is that right?
#
# Hint: write Bayes' theorem with the right symbols, then plug in. The
# denominator is P(T), which you computed marginally in Problem 3 of
# conditional_and_joint.py.


def analytic_disease_posterior() -> float:
    """Return P(D | T) for the disease/test setup above."""
    # TODO: implement Bayes' theorem.
    raise NotImplementedError()


def simulate_disease_posterior(n_trials: int = 500_000, rng=None) -> float:
    """Estimate P(D | T) from joint samples (D, T).

    Hint: among samples where the test is positive, what fraction have
    the disease? "Conditional probability via filtering" again - the same
    trick used in problem 1 of conditional_and_joint.py.
    """
    rng = rng or np.random.default_rng(0)
    # TODO: implement.
    raise NotImplementedError()


# ---------------------------------------------------------------------------
# Problem 2: sequential Bayesian updating
# ---------------------------------------------------------------------------
#
# Same disease, but the patient takes the test n times and each test
# comes back positive. After each test, your *posterior* becomes the
# *prior* for the next test (because the tests are conditionally
# independent given disease status).
#
# TASK
# ----
# Return a list whose k-th entry is P(D | T1, ..., Tk) - i.e. the
# probability the patient has the disease after k consecutive positives.
#
# What you should observe
# -----------------------
# - One positive: surprisingly small posterior.
# - Two positives: posterior jumps a lot.
# - Five positives: posterior pinned near 1.
#
# This is the *core* mechanism behind every Naive Bayes classifier
# accumulating evidence across tokens / features.


def sequential_disease_update(n_tests: int = 5) -> list[float]:
    """Return [P(D), P(D | T1), P(D | T1, T2), ...]."""
    # TODO: start with the prior, then loop n_tests times applying Bayes'
    # theorem with the *current* prior (= last posterior) each iteration.
    raise NotImplementedError()


# ---------------------------------------------------------------------------
# Problem 3: spam classifier (one-feature Naive Bayes)
# ---------------------------------------------------------------------------
#
# Two classes:  C in {spam, ham}, with P(spam) = 0.4.
# One feature:  word "free" is present (F=1) or absent (F=0), with
#               P(F=1 | spam) = 0.8 and P(F=1 | ham) = 0.1.
#
# TASK
# ----
# Compute P(spam | F=1).
#
# Hint: this is structurally identical to Problem 1 - just rename
# (D, T) to (spam, F=1). The fact that "the same formula solves both"
# is the point of Naive Bayes: features are different, the math is the
# same. Generalizing this to many words is left to the actual classifier
# scaffold under src/.


def analytic_spam_posterior() -> float:
    """Return P(spam | 'free' appears)."""
    # TODO: implement Bayes' theorem on the spam setup.
    raise NotImplementedError()


# ---------------------------------------------------------------------------


def main() -> None:
    print("=== Problem 1 - rare-disease test ===")
    print(f"analytic  P(D | T) = {analytic_disease_posterior():.5f}")
    print(f"simulated P(D | T) = {simulate_disease_posterior():.5f}")

    print("\n=== Problem 2 - sequential Bayesian updating ===")
    for k, p in enumerate(sequential_disease_update()):
        label = "prior" if k == 0 else f"after {k} positive test(s)"
        print(f"  {label:30s} P(D) = {p:.5f}")

    print("\n=== Problem 3 - spam classifier ===")
    print(f"  P(spam | 'free' appears) = {analytic_spam_posterior():.5f}")


if __name__ == "__main__":
    main()
