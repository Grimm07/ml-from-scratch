"""Ordinary least squares linear regression - placeholder.

Model
-----
    y_hat = X @ w + b      where X is (n_samples, n_features), w is (n_features,)
                            and b is a scalar bias.

Loss: mean squared error (MSE) =  (1/n) * sum_i (y_i - y_hat_i)^2.

Two implementation paths to consider
------------------------------------
1. Closed form (the "normal equation"):
       [w; b] = (X_aug.T @ X_aug)^-1 @ X_aug.T @ y
   where X_aug appends a column of ones to absorb the bias term.
   Pros: exact, one shot. Cons: O(d^3), unstable for ill-conditioned X.

2. Gradient descent on MSE:
       grad_w = -(2/n) * X.T @ (y - X @ w - b)
       grad_b = -(2/n) * sum(y - X @ w - b)
   Pros: scales to large d, generalizes to regularized variants.
   Cons: needs a learning rate; should record loss history for tests.

Things to research / decide
---------------------------
- np.linalg.lstsq vs np.linalg.solve vs forming the normal equation -
  which is most numerically stable, and why?
- Whether to store ``coef_`` and ``intercept_`` as separate attributes
  (sklearn convention) or a single augmented vector.
- How to detect "predict before fit" (raise vs return zeros) - the test
  in tests/test_supervised/test_linear_regression.py expects a raise.

The companion test scaffold is at
    tests/test_supervised/test_linear_regression.py
which expects ``fit(X, y)``, ``predict(X)``, ``coef_``, ``intercept_``.
"""


def fit():
    raise NotImplementedError()


def predict():
    raise NotImplementedError()
