"""Simulated annealing: probabilistic local search with a cooling schedule.

Signature you'll likely want
----------------------------
    def simulated_annealing(
        initial_state,
        objective,        # f(state) -> float, *minimized*
        neighbor,         # neighbor(state, rng) -> state
        n_steps=10_000,
        T0=1.0,
        cooling="geometric",  # alpha or callable T(step)
        alpha=0.999,
        rng=None,
    ) -> tuple[best_state, best_cost, history]

Algorithm sketch
----------------
    state = initial_state
    cost = objective(state)
    best = (state, cost)
    T = T0
    for step in range(n_steps):
        candidate = neighbor(state, rng)
        c = objective(candidate)
        dE = c - cost
        if dE <= 0 or rng.random() < exp(-dE / max(T, eps)):
            state, cost = candidate, c
            if cost < best[1]:
                best = (state, cost)
        T = update_temperature(T, step, alpha, cooling)
    return best, history

Things to think about while implementing
----------------------------------------
- *Always* track ``best`` separately from ``state`` - SA's current state
  can wander uphill, and you want the best state seen, not the last.
- Cooling schedules:
    * Geometric:    T = alpha * T          (alpha typically 0.99-0.9999)
    * Linear:       T = T0 - step * delta
    * Logarithmic:  T = T0 / log(step + 2) (theoretical guarantee, slow)
- ``exp(-dE / T)`` underflows when T is tiny - clamp T or guard with eps.
- For *maximization*, either negate the objective or flip the sign of dE.
- A worked exercise lives at problems/optimization/simulated_annealing.py
  using TSP as the example.

Connection to MCMC
------------------
Simulated annealing is Metropolis-Hastings with the temperature shrinking
to zero. At fixed T, the same algorithm samples from the Boltzmann
distribution exp(-cost(state) / T). That's why it's both an *optimizer*
(when T -> 0) and a *sampler* (when T is held fixed).
"""


def simulated_annealing():
    raise NotImplementedError()
