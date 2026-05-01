"""Tests for the from-scratch gradient boosting models.

Modules under test (paths anticipated; actual layout TBD):
    * ``src.supervised.ensemble.gradient_boosting_regressor``
    * ``src.supervised.ensemble.gradient_boosting_classifier``

Datasets
--------
* California Housing — regression baseline. GBM should comfortably beat the
  linear regression R^2 because it captures nonlinear feature interactions.
* Wine               — 3-class classification.
* Titanic (optional) — feature-engineering showcase. Loaded from disk if
  available under ``data/raw/titanic.csv`` and skipped otherwise so the test
  suite stays runnable on a fresh clone.

What these tests should cover once the models are implemented
-------------------------------------------------------------
1. Regressor:
   - Decreasing training loss as ``n_estimators`` grows (early trees help).
   - Beats a single decision-tree-stump baseline on California Housing.
   - ``feature_importances_`` is non-negative and sums to 1 (or to n_features
     if using raw gain - either convention is fine, just assert consistency).
2. Classifier:
   - Multiclass log-loss decreases monotonically across boosting rounds.
   - Achieves >= ~0.9 accuracy on Wine.
3. Determinism:
   - Same seed + same data => identical predictions.
4. Early stopping:
   - When provided a validation set, training halts at or before
     ``n_estimators`` and stores ``best_iteration_``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

gbm_reg = pytest.importorskip(
    "src.supervised.ensemble.gradient_boosting_regressor"
)
gbm_clf = pytest.importorskip(
    "src.supervised.ensemble.gradient_boosting_classifier"
)


# ---------------------------------------------------------------------------
# Regression on California Housing
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="GBM regressor not implemented", strict=False)
def test_california_housing_beats_linear_baseline(california_housing, random_seed):
    """GBM should achieve R^2 >= 0.75 on a 80/20 split (linear OLS sits ~0.6)."""
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    X, y = california_housing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed
    )
    model = gbm_reg.GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_seed
    )
    model.fit(X_train, y_train)
    assert r2_score(y_test, model.predict(X_test)) >= 0.75


@pytest.mark.xfail(reason="GBM regressor not implemented", strict=False)
def test_training_loss_decreases(california_housing, random_seed):
    """The training loss recorded after each boosting round must be non-increasing."""
    X, y = california_housing
    model = gbm_reg.GradientBoostingRegressor(
        n_estimators=20, learning_rate=0.1, random_state=random_seed
    )
    model.fit(X, y)
    history = np.asarray(model.train_loss_)
    assert np.all(np.diff(history) <= 1e-8)


@pytest.mark.xfail(reason="GBM regressor not implemented", strict=False)
def test_determinism_under_fixed_seed(california_housing, random_seed):
    """Two fits with the same seed must produce identical predictions."""
    X, y = california_housing
    a = gbm_reg.GradientBoostingRegressor(n_estimators=10, random_state=random_seed)
    b = gbm_reg.GradientBoostingRegressor(n_estimators=10, random_state=random_seed)
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


# ---------------------------------------------------------------------------
# Classification on Wine
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="GBM classifier not implemented", strict=False)
def test_wine_multiclass_accuracy(wine, random_seed):
    """Wine should be classified at >= 90% on a held-out split."""
    from sklearn.model_selection import train_test_split

    X, y = wine
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_seed
    )
    model = gbm_clf.GradientBoostingClassifier(
        n_estimators=50, learning_rate=0.1, max_depth=3, random_state=random_seed
    )
    model.fit(X_train, y_train)
    assert (model.predict(X_test) == y_test).mean() >= 0.9


# ---------------------------------------------------------------------------
# Optional: Titanic (feature engineering showcase)
# ---------------------------------------------------------------------------


TITANIC_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "titanic.csv"


@pytest.mark.skipif(
    not TITANIC_PATH.exists(),
    reason="Optional dataset; place titanic.csv under data/raw/ to enable.",
)
@pytest.mark.xfail(reason="GBM classifier not implemented", strict=False)
def test_titanic_survival_classifier(random_seed):
    """End-to-end smoke test on Titanic: load, encode, fit, score >= 0.78 AUC.

    Intentionally light on feature engineering - the goal is to verify the
    classifier handles a small mixed-type tabular problem, not to win Kaggle.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(TITANIC_PATH)
    y = df["Survived"].to_numpy()
    X = pd.get_dummies(
        df.drop(columns=["Survived", "Name", "Ticket", "Cabin"]),
        drop_first=True,
    ).fillna(0.0).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_seed
    )
    model = gbm_clf.GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_seed
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    assert roc_auc_score(y_test, proba) >= 0.78
