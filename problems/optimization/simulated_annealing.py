"""Simulated annealing on a small Travelling Salesman Problem.

Why TSP?
--------
TSP is the canonical "rugged landscape" toy problem: many local optima,
no useful gradient, and a tiny instance fits on a screen. It's a clean
showcase for why simulated annealing exists at all - greedy 2-opt gets
stuck, SA escapes by occasionally accepting *worse* moves.

The algorithm
-------------
At step t with current solution x_t:
    1. Propose a neighbor x' (here: reverse a random sub-tour - the
       standard "2-opt" move).
    2. Compute dE = cost(x') - cost(x_t).
    3. If dE <= 0  -> always accept.
       Otherwise   -> accept with probability exp(-dE / T_t).
    4. Cool: T_{t+1} = alpha * T_t with alpha slightly below 1.

Notation: T is "temperature". High T  => almost any move is accepted
(exploration). Low T => only improvements are accepted (exploitation).
The geometric schedule T <- alpha * T is the simplest cooling rule;
others (logarithmic, linear) have different theoretical guarantees.

Why this matters for ML
-----------------------
- Hyperparameter and architecture search use SA-like methods.
- Discrete-decision problems in ML pipelines (feature subset selection,
  hard-assignment clustering) are non-differentiable, so SGD doesn't
  apply - SA gives you something to do instead.
- The "accept worse moves with some probability" idea reappears in MCMC
  (Metropolis-Hastings), where instead of minimizing it samples from a
  distribution. Notice the symmetry: SA is MCMC where the temperature
  is shrinking to zero.

Things to research while solving
--------------------------------
- "Detailed balance" in Metropolis-Hastings - why exp(-dE/T) is the
  right acceptance probability.
- Cooling schedules and their convergence guarantees (Kirkpatrick 1983,
  Geman & Geman 1984).
- 2-opt vs 3-opt for TSP neighborhoods - what tradeoffs?

What to look for when you've implemented it
-------------------------------------------
- Greedy hill-climbing usually gets stuck; SA wins on average across
  many random initializations on a 30-city instance.
- The accepted-cost trace is non-monotonic early (escaping local optima)
  and monotonic later (converged).
- Cooling too fast -> trapped at a local optimum.
- Cooling too slow -> wasted compute.
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Problem instance: random cities in the unit square.
# ---------------------------------------------------------------------------


def random_cities(n: int = 30, rng=None) -> np.ndarray:
    """Return an (n, 2) array of cities in the unit square."""
    rng = rng or np.random.default_rng(0)
    return rng.uniform(0, 1, size=(n, 2))


def tour_length(tour: np.ndarray, cities: np.ndarray) -> float:
    """Total length of a *closed* tour - sum of edge distances, wrap to start.

    Hint: indexing cities[tour] gives the coordinate sequence; np.diff
    with `append` lets you compute consecutive distances in one shot.
    """
    # TODO: implement. Should return a float, and a permutation of
    # [0..n-1] should give the same length regardless of starting index
    # (it's a cycle).
    raise NotImplementedError()


# ---------------------------------------------------------------------------
# Neighbor proposal: 2-opt reversal of a random sub-tour.
# ---------------------------------------------------------------------------


def two_opt_neighbor(tour: np.ndarray, rng) -> np.ndarray:
    """Pick a random pair of indices (i, j) and reverse tour[i:j+1].

    Why this works: 2-opt removes two edges and reconnects them in the
    other valid way. It's the smallest non-trivial change to a tour.

    Hint: be careful not to mutate the input - return a copy.
    """
    # TODO: implement.
    raise NotImplementedError()


# ---------------------------------------------------------------------------
# Greedy baseline: only accept improvements (T = 0).
# ---------------------------------------------------------------------------


def greedy_two_opt(
    cities: np.ndarray,
    n_steps: int,
    rng,
) -> tuple[np.ndarray, float]:
    """Random-restart 2-opt that only accepts improvements.

    Hint: start from a random permutation, propose 2-opt neighbors, only
    accept when cost strictly decreases. This will get stuck at local
    optima - that's the *point* of the comparison with SA.
    """
    # TODO: implement.
    raise NotImplementedError()


# ---------------------------------------------------------------------------
# Simulated annealing.
# ---------------------------------------------------------------------------


def simulated_annealing(
    cities: np.ndarray,
    n_steps: int = 20_000,
    T0: float = 1.0,
    alpha: float = 0.9995,
    rng=None,
) -> tuple[np.ndarray, float]:
    """Standard SA with a geometric cooling schedule.

    Skeleton
    --------
        tour = random permutation
        cost = tour_length(tour, cities)
        best_tour, best_cost = tour, cost
        T = T0
        for _ in range(n_steps):
            candidate = two_opt_neighbor(tour, rng)
            c = tour_length(candidate, cities)
            dE = c - cost
            if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-12)):
                tour, cost = candidate, c
                if cost < best_cost:
                    best_tour, best_cost = tour.copy(), cost
            T *= alpha
        return best_tour, best_cost

    The skeleton is *deliberately* shown - the lesson here is the
    *structure* of the algorithm. You'll still need to fill it in,
    pick the right RNG calls, decide on copy semantics, etc.
    """
    rng = rng or np.random.default_rng(0)
    # TODO: implement using the skeleton in the docstring.
    raise NotImplementedError()


# ---------------------------------------------------------------------------


def main() -> None:
    """Compare greedy 2-opt and SA on a 30-city TSP, averaged over seeds.

    The single-seed run can tie (SA and greedy can both find the optimum
    on tiny instances given enough proposals). Averaging over many random
    seeds is what reveals SA's edge.
    """
    cities = random_cities(n=30, rng=np.random.default_rng(0))
    n_runs = 20
    greedy_costs, sa_costs = [], []
    for seed in range(n_runs):
        rng = np.random.default_rng(seed)
        _, gc = greedy_two_opt(cities, n_steps=2_000, rng=rng)
        rng = np.random.default_rng(seed)
        _, sc = simulated_annealing(cities, n_steps=20_000, rng=rng)
        greedy_costs.append(gc)
        sa_costs.append(sc)

    print(f"Average over {n_runs} initializations on 30-city TSP:")
    print(f"  Greedy 2-opt mean tour length:    {np.mean(greedy_costs):.4f}")
    print(f"  Simulated annealing mean length:  {np.mean(sa_costs):.4f}")
    print(
        f"  SA win rate vs greedy:            "
        f"{(np.array(sa_costs) < np.array(greedy_costs)).mean():.0%}"
    )
    print(
        "\nThings to try once it works:"
        "\n  * Sweep alpha in [0.99, 0.999, 0.9999] - cooling speed matters."
        "\n  * Sweep T0 in [0.01, 0.1, 1.0, 10.0] - too cold and SA is greedy;"
        "\n    too hot and SA is a random walk."
        "\n  * Replace 2-opt with random swaps - watch quality drop."
    )


if __name__ == "__main__":
    main()
