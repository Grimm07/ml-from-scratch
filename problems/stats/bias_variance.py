"""Bias-variance decomposition demonstrated on polynomial regression.

The decomposition
-----------------
For squared-error loss, the expected error of a model f_hat at a point x
decomposes into

    E[(y - f_hat(x))^2] = Bias[f_hat(x)]^2 + Var[f_hat(x)] + sigma^2

where the expectation is over both the training set and the noise, and
sigma^2 is the irreducible noise variance.

* Bias       systematic error - how far the *average* prediction (over
             many training sets) is from the truth. Underfitting models
             have high bias.
* Variance   how much the prediction wobbles when the training set
             changes. Overfitting models have high variance.
* Noise      the floor; can't be removed without more / less-noisy data.

Strategy
--------
Sample many independent training sets from the same data-generating
process. Fit a polynomial of degree d to each. Then for each test point,
look at the *ensemble* of predictions and compute:

    bias^2(x)   = (mean_predictions(x) - truth(x))^2
    variance(x) = empirical variance of predictions(x) across datasets

Average over test points, repeat for several degrees d, and plot. The
total error vs degree should be U-shaped: high bias on the left, high
variance on the right, sweet spot somewhere in the middle.

Why this matters for ML
-----------------------
Every regularization knob (model size, weight decay, dropout, early
stopping, data augmentation) trades variance for bias. Knowing which
side of the U you're on tells you which knob to turn.

Things to research
------------------
- Why np.polyfit warns / blows up at very high degree on small data.
- "Double descent" - the modern phenomenon where the U becomes a UU.
- Why this decomposition only cleanly applies to squared loss.
"""

from __future__ import annotations

import numpy as np


def true_function(x: np.ndarray) -> np.ndarray:
    """The underlying signal we are trying to learn (a sine wave)."""
    return np.sin(2 * np.pi * x)


def sample_training_set(n: int, noise_sigma: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Draw n iid (x, y) pairs with x ~ Uniform(0,1), y = sin(2*pi*x) + noise."""
    # TODO: sample x uniformly, compute y = true_function(x) + Gaussian noise.
    raise NotImplementedError()


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Return polynomial coefficients fit by least squares.

    Hint: numpy has np.polyfit / np.polyval. Highest degree first by
    convention.
    """
    # TODO: one-liner; the lesson is in the surrounding analysis, not here.
    raise NotImplementedError()


def decompose(
    degrees: list[int],
    n_train: int = 30,
    n_datasets: int = 200,
    noise_sigma: float = 0.3,
):
    """Return {degree: {'bias^2', 'variance', 'noise^2', 'total'}}.

    Hints
    -----
    * Sample n_datasets *independent* training sets and fit a degree-d
      polynomial to each.
    * Predict on a fixed grid of test x-values (50 points is plenty).
    * predictions[k, i] = prediction of model k at test point i.
    * bias^2 at x_i  = (predictions[:, i].mean() - truth(x_i))^2
    * variance at x_i = predictions[:, i].var()
    * Average across i to get scalar bias^2 and variance for this degree.
    * total = bias^2 + variance + noise_sigma^2.
    """
    # TODO: implement using the hints above.
    raise NotImplementedError()


def main() -> None:
    degrees = [0, 1, 3, 5, 9, 15]
    table = decompose(degrees)
    header = f"{'degree':>6}  {'bias^2':>10}  {'variance':>10}  {'noise^2':>10}  {'total':>10}"
    print(header)
    print("-" * len(header))
    for d in degrees:
        r = table[d]
        print(
            f"{d:>6}  {r['bias^2']:>10.4f}  {r['variance']:>10.4f}  "
            f"{r['noise^2']:>10.4f}  {r['total']:>10.4f}"
        )
    print(
        "\nLow-degree rows should have high bias^2 and low variance."
        "\nHigh-degree rows should explode in variance."
        "\nThe minimum-total row is the sweet spot for *this* problem and *this* n."
    )


if __name__ == "__main__":
    main()
