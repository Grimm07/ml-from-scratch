"""Tests for the from-scratch linear regression model.

Module under test: ``src.supervised.linear_models.linear_regression``

Dataset
-------
Uses the California Housing regression dataset from scikit-learn
(20,640 samples, 8 numeric features, target = median house value).
This is the canonical regression benchmark for the project, chosen
because it is large enough to expose numerical issues yet small
enough to fit comfortably in memory.

What these tests should cover once the model is implemented
-----------------------------------------------------------
1. Shape contracts:
   - ``fit(X, y)`` accepts (n_samples, n_features) and (n_samples,)
   - ``predict(X)`` returns (n_samples,)
2. Fit produces a coefficient vector of length n_features and a scalar bias.
3. On a tiny linearly-separable problem (y = X @ true_w + true_b) the learned
   coefficients should match the analytic solution within float tolerance.
4. On California Housing, R^2 on a held-out split should be >= ~0.55, which
   is the rough OLS baseline. This guards against silent regressions that
   still happen to "fit" on toy data.
5. Predicting before fit should raise (or return a documented sentinel).
6. Re-fitting on new data should reset state (no leakage from prior fit).
"""

from __future__ import annotations

import numpy as np
import pytest

# ``importorskip`` keeps test collection green while the model is still a
# placeholder. Once ``fit`` and ``predict`` are real, the skips disappear.
linear_regression = pytest.importorskip(
    "src.supervised.linear_models.linear_regression"
)


# ---------------------------------------------------------------------------
# Smoke tests on a synthetic, perfectly linear problem
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_linear(random_seed: int):
    """Generate a noiseless linear problem with known coefficients.

    Used to verify the optimizer recovers the ground truth (analytical OLS
    solution) to high precision.
    """
    rng = np.random.default_rng(random_seed)
    n_samples, n_features = 200, 5
    X = rng.standard_normal((n_samples, n_features))
    true_w = np.array([1.5, -2.0, 0.5, 0.0, 3.25])
    true_b = 0.75
    y = X @ true_w + true_b
    return X, y, true_w, true_b


@pytest.mark.xfail(reason="fit/predict are not implemented yet", strict=False)
def test_fit_recovers_known_coefficients(synthetic_linear):
    """The learned weights should match the ground truth on a noiseless problem."""
    X, y, true_w, true_b = synthetic_linear
    model = linear_regression  # replace with a model class once one exists
    model.fit(X, y)
    # Replace these attribute names if the implementation uses different ones.
    np.testing.assert_allclose(model.coef_, true_w, atol=1e-6)
    np.testing.assert_allclose(model.intercept_, true_b, atol=1e-6)


@pytest.mark.xfail(reason="fit/predict are not implemented yet", strict=False)
def test_predict_shape_matches_input(synthetic_linear):
    """``predict`` should return one prediction per input row."""
    X, y, _, _ = synthetic_linear
    linear_regression.fit(X, y)
    y_hat = linear_regression.predict(X)
    assert y_hat.shape == y.shape


@pytest.mark.xfail(reason="fit/predict are not implemented yet", strict=False)
def test_predict_before_fit_raises():
    """Calling ``predict`` before ``fit`` is a programmer error and should raise."""
    with pytest.raises((RuntimeError, ValueError, AttributeError)):
        linear_regression.predict(np.zeros((1, 5)))


# ---------------------------------------------------------------------------
# Real-data baseline on California Housing
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="fit/predict are not implemented yet", strict=False)
def test_california_housing_r2_baseline(california_housing, random_seed: int):
    """Trained on California Housing, R^2 on a held-out split should beat ~0.55."""
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    X, y = california_housing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed
    )
    linear_regression.fit(X_train, y_train)
    y_hat = linear_regression.predict(X_test)
    assert r2_score(y_test, y_hat) >= 0.55
