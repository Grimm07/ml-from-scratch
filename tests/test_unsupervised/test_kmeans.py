"""Tests for the from-scratch K-Means clustering algorithm.

Module under test: ``src.unsupervised.clustering.kmeans`` (path TBD).

Datasets
--------
* ``make_blobs``  — 4 well-separated isotropic Gaussians. With k=4 a correct
  implementation should recover the ground-truth labels almost perfectly
  (Adjusted Rand Index >= 0.95).
* Iris            — 3 species; not perfectly separable in feature space, so
  the bar is lower (ARI >= 0.6 is sklearn's typical k-means score).
* Digits          — 64-D, 10 clusters. Stress-test of the algorithm in higher
  dimensions; only checks that inertia is finite and labels are valid.

What these tests should cover once the model is implemented
-----------------------------------------------------------
1. API contract:
   - ``fit(X)`` populates ``cluster_centers_`` of shape (k, n_features).
   - ``labels_`` has shape (n_samples,) with integer values in [0, k).
   - ``predict(X_new)`` returns the index of the nearest centroid.
2. Convergence:
   - Inertia (sum of squared distances to nearest centroid) is non-increasing
     across Lloyd's iterations.
   - Algorithm stops within ``max_iter`` and reports ``n_iter_``.
3. k-means++ init:
   - Initial centroids are distinct rows of ``X``.
   - With the same seed, init produces deterministic centroids.
4. Quality:
   - On make_blobs(centers=4), ARI vs. ground truth >= 0.95.
   - On Iris, ARI vs. true species >= 0.6.
"""

from __future__ import annotations

import numpy as np
import pytest

kmeans = pytest.importorskip("src.unsupervised.clustering.kmeans")


# ---------------------------------------------------------------------------
# API and convergence
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="K-Means not implemented", strict=False)
def test_fit_populates_attributes(blobs, random_seed):
    """After fit, the model exposes centers, labels, inertia and iteration count."""
    X, _ = blobs
    model = kmeans.KMeans(n_clusters=4, random_state=random_seed)
    model.fit(X)
    assert model.cluster_centers_.shape == (4, X.shape[1])
    assert model.labels_.shape == (X.shape[0],)
    assert set(np.unique(model.labels_)).issubset(set(range(4)))
    assert model.inertia_ >= 0.0
    assert model.n_iter_ >= 1


@pytest.mark.xfail(reason="K-Means not implemented", strict=False)
def test_inertia_history_is_non_increasing(blobs, random_seed):
    """Lloyd's algorithm is guaranteed to be monotonically non-increasing."""
    X, _ = blobs
    model = kmeans.KMeans(n_clusters=4, random_state=random_seed, record_history=True)
    model.fit(X)
    history = np.asarray(model.inertia_history_)
    assert np.all(np.diff(history) <= 1e-9)


# ---------------------------------------------------------------------------
# Quality on real / synthetic data
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="K-Means not implemented", strict=False)
def test_blobs_recovers_ground_truth(blobs, random_seed):
    """ARI vs. true cluster labels should be >= 0.95 on well-separated blobs."""
    from sklearn.metrics import adjusted_rand_score

    X, y = blobs
    model = kmeans.KMeans(n_clusters=4, random_state=random_seed, n_init=10)
    model.fit(X)
    assert adjusted_rand_score(y, model.labels_) >= 0.95


@pytest.mark.xfail(reason="K-Means not implemented", strict=False)
def test_iris_clusters_align_with_species(iris, random_seed):
    """Iris is not perfectly separable; aim for ARI >= 0.6 (sklearn's range)."""
    from sklearn.metrics import adjusted_rand_score

    X, y = iris
    model = kmeans.KMeans(n_clusters=3, random_state=random_seed, n_init=10)
    model.fit(X)
    assert adjusted_rand_score(y, model.labels_) >= 0.6


@pytest.mark.xfail(reason="K-Means not implemented", strict=False)
def test_digits_clustering_smoke(digits, random_seed):
    """Higher-dimensional smoke test: K-Means should produce valid labels and
    finite inertia on the 64-D digits dataset.
    """
    X, _ = digits
    model = kmeans.KMeans(n_clusters=10, random_state=random_seed, n_init=5)
    model.fit(X)
    assert np.isfinite(model.inertia_)
    assert model.labels_.shape == (X.shape[0],)


# ---------------------------------------------------------------------------
# k-means++ initialization
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="K-Means not implemented", strict=False)
def test_kmeans_pp_init_picks_distinct_points(blobs, random_seed):
    """k-means++ should select k distinct rows of X as the initial centroids."""
    X, _ = blobs
    init = kmeans.kmeans_pp_init(X, n_clusters=4, random_state=random_seed)
    assert init.shape == (4, X.shape[1])
    assert len({tuple(row) for row in init}) == 4


@pytest.mark.xfail(reason="K-Means not implemented", strict=False)
def test_predict_on_new_data(blobs, random_seed):
    """``predict`` on the training data must agree with ``labels_``."""
    X, _ = blobs
    model = kmeans.KMeans(n_clusters=4, random_state=random_seed)
    model.fit(X)
    np.testing.assert_array_equal(model.predict(X), model.labels_)
