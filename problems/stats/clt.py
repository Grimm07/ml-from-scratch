"""Central Limit Theorem - the reason "everything looks Gaussian eventually".

Statement
---------
Let X_1, ..., X_n be iid with mean mu and finite variance sigma^2. Define
the sample mean

    X_bar_n = (1/n) sum_{i=1..n} X_i.

Then as n -> infinity,

    sqrt(n) * (X_bar_n - mu) -->  Normal(0, sigma^2)   in distribution.

In words: regardless of the source distribution (uniform, exponential,
Bernoulli, ...), the *distribution of the sample mean* tends to a
Gaussian as n grows. The variance of the sample mean shrinks like
sigma^2 / n.

Why this matters for ML
-----------------------
- Confidence intervals on metrics computed via averaging (accuracy,
  log-loss, MSE) are Gaussian-shaped for large enough test sets.
- A mini-batch gradient in SGD is approximately Gaussian around the
  full-batch gradient by exactly this argument.
- The Gaussian noise assumption in linear regression is justified
  post-hoc via the CLT applied to whatever residual-generating process
  produced the noise.

Things to research
------------------
- The standard error: SE(X_bar) = sigma / sqrt(n). Why "sqrt(n)"?
- Skew and excess kurtosis of common distributions (exp, uniform, lognormal).
- When the CLT *fails*: heavy-tailed distributions (Cauchy, infinite variance).

Strategy
--------
The script below pulls iid samples from a *non-Gaussian* source (the
exponential distribution is a good choice because it's heavily right-
skewed) and looks at how the distribution of the sample mean behaves as
the sample size n grows. As n increases:

* The empirical *std* of the sample mean should track sigma / sqrt(n).
* The empirical *skew* should drift toward 0 (Gaussian is symmetric).
* The empirical *excess kurtosis* should drift toward 0.
"""

from __future__ import annotations

import numpy as np


def sample_means_from_exponential(
    n: int,
    n_trials: int = 50_000,
    rate: float = 1.0,
    rng=None,
) -> np.ndarray:
    """Return n_trials sample means, each computed from n exponential draws.

    Hints
    -----
    * np.random.Generator.exponential takes a `scale` parameter (1/rate).
    * Build a (n_trials, n) matrix and average along axis=1; vectorize.
    """
    rng = rng or np.random.default_rng(0)
    # TODO: implement.
    raise NotImplementedError()


def summarize(means: np.ndarray) -> dict[str, float]:
    """Return mean, std, skew, and excess kurtosis of an empirical sample.

    Hints
    -----
    * Skew is the third standardized moment: E[((X - mu)/sigma)^3].
    * Excess kurtosis is the fourth standardized moment minus 3
      (so a Gaussian has excess kurtosis 0).
    * scipy.stats.skew / kurtosis exist if you want a sanity check, but
      derive these by hand first - it'll cement the definitions.
    """
    # TODO: implement.
    raise NotImplementedError()


def main() -> None:
    print("Source: Exponential(1) -- mean=1, std=1, skew=2, excess kurtosis=6")
    print(f"{'n':>6}  {'mean':>8}  {'std':>8}  {'skew':>8}  {'excess kurt':>12}")
    for n in (1, 2, 5, 30, 100, 1_000):
        m = sample_means_from_exponential(n)
        s = summarize(m)
        print(
            f"{n:>6}  {s['mean']:>8.4f}  {s['std']:>8.4f}  "
            f"{s['skew']:>8.4f}  {s['excess_kurt']:>12.4f}"
        )
    print(
        "\nWhat to look for:"
        "\n  * std should follow 1/sqrt(n)  (since the source has sigma = 1)."
        "\n  * skew and excess kurtosis should approach 0 as n grows."
        "\n  * If they don't, your source distribution might have infinite variance"
        "\n    (try Cauchy and watch the CLT fail spectacularly)."
    )


if __name__ == "__main__":
    main()
