"""Tests for the from-scratch logistic regression classifier.

Module under test: ``src.supervised.linear_models.logistic_regression``
(currently not yet scaffolded - tests will be skipped until it exists).

Datasets
--------
* Iris  — 150 samples, 4 features, 3 balanced classes. Used for the multiclass
  path (one-vs-rest or softmax depending on implementation).
* Wine  — 178 samples, 13 features, 3 classes. Higher-dimensional sanity check
  that also exercises feature scaling assumptions.
* A synthetic 2D two-class blob is used for the binary smoke test so that we
  can assert the decision boundary by hand.

What these tests should cover once the model is implemented
-----------------------------------------------------------
1. Binary path:
   - ``predict_proba`` outputs probabilities in [0, 1] that sum to 1 per row.
   - ``predict`` returns the argmax class.
   - Achieves >= 95% accuracy on a clearly linearly-separable 2D problem.
2. Multiclass path (Iris/Wine):
   - Output shape is (n_samples, n_classes).
   - Each row of ``predict_proba`` sums to 1 within float tolerance.
   - Achieves >= ~0.9 accuracy on Iris (sklearn's logistic regression hits
     ~0.97; we leave headroom for hand-rolled optimizers).
3. Convergence / regularization:
   - With high L2 regularization, weights stay small.
   - Loss is monotonically non-increasing over training iterations.
"""

from __future__ import annotations

import numpy as np
import pytest

logistic_regression = pytest.importorskip(
    "src.supervised.linear_models.logistic_regression"
)


# ---------------------------------------------------------------------------
# Binary classification - synthetic
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_blobs(random_seed: int):
    """Two well-separated 2D Gaussian blobs - linearly separable.

    Used to assert the binary classifier reaches near-perfect accuracy and
    that ``predict_proba`` is well-calibrated near the decision boundary.
    """
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        n_samples=400,
        centers=2,
        cluster_std=1.0,
        n_features=2,
        random_state=random_seed,
    )
    return X, y


@pytest.mark.xfail(reason="logistic regression not implemented", strict=False)
def test_binary_predict_proba_is_valid_distribution(binary_blobs):
    """Probabilities should be in [0, 1] and sum to 1 across the two classes."""
    X, y = binary_blobs
    logistic_regression.fit(X, y)
    proba = logistic_regression.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.xfail(reason="logistic regression not implemented", strict=False)
def test_binary_accuracy_on_separable_data(binary_blobs):
    """Linearly separable blobs should be classified at >= 95% accuracy."""
    X, y = binary_blobs
    logistic_regression.fit(X, y)
    y_hat = logistic_regression.predict(X)
    assert (y_hat == y).mean() >= 0.95


# ---------------------------------------------------------------------------
# Multiclass classification - Iris and Wine
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="logistic regression not implemented", strict=False)
def test_iris_multiclass_accuracy(iris, random_seed: int):
    """Iris should be classified at >= 90% accuracy on a held-out split."""
    from sklearn.model_selection import train_test_split

    X, y = iris
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_seed
    )
    logistic_regression.fit(X_train, y_train)
    assert (logistic_regression.predict(X_test) == y_test).mean() >= 0.9


@pytest.mark.xfail(reason="logistic regression not implemented", strict=False)
def test_wine_predict_proba_sums_to_one(wine):
    """For 3-class Wine, predict_proba rows must sum to 1."""
    X, y = wine
    logistic_regression.fit(X, y)
    proba = logistic_regression.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Optimization properties
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="logistic regression not implemented", strict=False)
def test_loss_is_non_increasing(binary_blobs):
    """Training loss history should be monotonically non-increasing.

    A monotonic loss is a strong signal that the gradient and learning-rate
    logic are correct. Allow tiny ULP-scale jumps via ``atol``.
    """
    X, y = binary_blobs
    logistic_regression.fit(X, y)
    history = np.asarray(logistic_regression.loss_history_)
    diffs = np.diff(history)
    assert np.all(diffs <= 1e-8), "loss increased during training"
