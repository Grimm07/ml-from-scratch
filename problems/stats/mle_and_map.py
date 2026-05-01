"""Maximum Likelihood and Maximum A Posteriori estimation for a Bernoulli.

Maximum Likelihood Estimation (MLE)
-----------------------------------
Given iid coin flips x_1, ..., x_n with x_i in {0, 1}, the likelihood
under Bernoulli(theta) is

    L(theta) = prod_i theta^x_i * (1 - theta)^(1 - x_i).

Take logs, differentiate w.r.t. theta, set to zero. The MLE comes out
to a familiar quantity. Derive it - don't just look it up.

Maximum A Posteriori (MAP)
--------------------------
Treat theta as a random variable with prior p(theta) = Beta(alpha, beta).
This prior encodes the belief that theta is around alpha / (alpha + beta)
with strength roughly alpha + beta - 2 "pseudo-counts" (i.e. the prior
"feels like" you've already seen alpha - 1 successes and beta - 1
failures before the data).

Bayes:    p(theta | data)  ~  p(data | theta) * p(theta).

The Beta is the *conjugate prior* for the Bernoulli - the posterior is
itself a Beta. Find its parameters in terms of (alpha, beta, n, s) where
s = number of successes, then take the *mode* of that Beta to get the
MAP estimator.

Why this matters for ML
-----------------------
- L2 regularization in linear regression IS MAP under a Gaussian weight prior.
- L1 regularization IS MAP under a Laplace weight prior.
- "Laplace smoothing" in Naive Bayes (the +1) IS MAP under Beta(1,1)
  / Dirichlet(1, ..., 1).
- As n grows, the prior is washed out and theta_MAP -> theta_MLE - i.e.
  regularization matters most when data is scarce, exactly when
  overfitting is most dangerous.

Things to research while solving
--------------------------------
- "Conjugate prior" - why Beta + Bernoulli closes nicely.
- "Mode of Beta(a, b)" - slightly different from the mean; needed for MAP.
- What happens to theta_MAP as alpha -> 1, beta -> 1? What does that
  prior represent? (Answer: uniform on [0,1]; MAP collapses to MLE.)
"""

from __future__ import annotations

import numpy as np


def estimate(samples: np.ndarray, alpha: float = 2.0, beta: float = 2.0):
    """Return (theta_MLE, theta_MAP) for the Bernoulli given iid 0/1 samples.

    Parameters
    ----------
    samples : array of 0s and 1s.
    alpha, beta : Beta prior hyperparameters. Beta(2, 2) is a weak prior
        centered at 0.5 worth ~2 pseudo-flips.

    Hints
    -----
    * theta_MLE: count successes, divide by n.
    * theta_MAP: figure out the posterior Beta(?, ?) parameters in terms
      of (alpha, beta, s, n - s), then take the mode formula
      mode(Beta(a, b)) = (a - 1) / (a + b - 2)   for a, b > 1.
    """
    # TODO: implement both estimators.
    raise NotImplementedError()


def main() -> None:
    rng = np.random.default_rng(0)
    true_theta = 0.3
    alpha, beta = 2.0, 2.0

    print(f"True theta = {true_theta}")
    print(f"Prior      = Beta({alpha}, {beta})\n")
    print(f"{'n':>6}  {'theta_MLE':>10}  {'theta_MAP':>10}")
    for n in (5, 20, 100, 1_000, 10_000):
        samples = (rng.random(n) < true_theta).astype(int)
        mle, map_ = estimate(samples, alpha=alpha, beta=beta)
        print(f"{n:>6}  {mle:>10.4f}  {map_:>10.4f}")
    print(
        "\nWatch how theta_MAP converges to theta_MLE as n grows."
        "\nAt small n, theta_MAP is pulled toward the prior's mean (0.5)."
    )


if __name__ == "__main__":
    main()
