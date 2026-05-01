"""Tests for the from-scratch PCA implementation.

Module under test: ``src.unsupervised.dimensionality_reduction.pca`` (path TBD).

Datasets
--------
* Iris   — 4 features, 3 classes. Standard low-dimensional sanity check; the
  first two principal components should already separate the classes well.
* Digits — 64 features, 10 classes. Used to verify PCA scales and that the
  components capture the expected fraction of variance (>= 50% in 8 PCs).
* A synthetic dataset with a known low-rank structure is used to assert that
  PCA recovers the planted basis (up to a sign/permutation).

What these tests should cover once the model is implemented
-----------------------------------------------------------
1. API contract:
   - ``fit(X)`` populates ``components_`` of shape (n_components, n_features).
   - ``explained_variance_ratio_`` is non-negative and sums to <= 1.
   - ``transform(X)`` returns shape (n_samples, n_components).
   - ``inverse_transform(transform(X))`` reconstructs X with bounded error.
2. Mathematical correctness:
   - Components are orthonormal (rows of ``components_`` are unit length and
     mutually orthogonal).
   - Explained variances are in non-increasing order.
   - On a synthetic rank-2 problem, the top 2 components span the planted
     subspace (subspace angle ~ 0).
3. Centering:
   - PCA on data with non-zero mean produces the same components as PCA on
     centered data (i.e. the implementation centers internally).
"""

from __future__ import annotations

import numpy as np
import pytest

pca = pytest.importorskip("src.unsupervised.dimensionality_reduction.pca")


# ---------------------------------------------------------------------------
# API and basic invariants
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="PCA not implemented", strict=False)
def test_components_shape_and_orthonormality(iris):
    """Components should be orthonormal: C @ C.T == I_k."""
    X, _ = iris
    model = pca.PCA(n_components=2)
    model.fit(X)
    C = model.components_
    assert C.shape == (2, X.shape[1])
    np.testing.assert_allclose(C @ C.T, np.eye(2), atol=1e-6)


@pytest.mark.xfail(reason="PCA not implemented", strict=False)
def test_explained_variance_ordering_and_bounds(digits):
    """Explained variance ratios are non-negative, non-increasing, and sum <= 1."""
    X, _ = digits
    model = pca.PCA(n_components=8)
    model.fit(X)
    ratios = model.explained_variance_ratio_
    assert np.all(ratios >= 0.0)
    assert np.all(np.diff(ratios) <= 1e-12)
    assert ratios.sum() <= 1.0 + 1e-9


@pytest.mark.xfail(reason="PCA not implemented", strict=False)
def test_transform_inverse_reconstruction(iris):
    """X ~= inverse_transform(transform(X)) when n_components == n_features."""
    X, _ = iris
    model = pca.PCA(n_components=X.shape[1])
    model.fit(X)
    X_round = model.inverse_transform(model.transform(X))
    np.testing.assert_allclose(X_round, X, atol=1e-6)


# ---------------------------------------------------------------------------
# Recovering planted structure
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="PCA not implemented", strict=False)
def test_recovers_low_rank_subspace(random_seed):
    """On a rank-2 synthetic problem, the top 2 PCs span the planted subspace.

    We construct ``X = Z @ B`` where Z is (n, 2) Gaussian and B is a fixed
    (2, 5) basis. The 2-D principal subspace of X must equal range(B^T).
    Subspace agreement is measured via the largest principal angle - if it's
    close to 0 the subspaces match (up to rotation).
    """
    rng = np.random.default_rng(random_seed)
    Z = rng.standard_normal((500, 2))
    B = rng.standard_normal((2, 5))
    X = Z @ B  # samples lie in the 2-D row space of B

    model = pca.PCA(n_components=2)
    model.fit(X)

    # Largest principal angle between the two subspaces.
    Q1, _ = np.linalg.qr(model.components_.T)
    Q2, _ = np.linalg.qr(B.T)
    sing = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    largest_angle = np.arccos(np.clip(sing.min(), -1.0, 1.0))
    assert largest_angle < 1e-3


# ---------------------------------------------------------------------------
# Centering behavior
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="PCA not implemented", strict=False)
def test_centering_is_invariant_to_translation(iris):
    """Adding a constant offset to X must not change the principal axes."""
    X, _ = iris

    a = pca.PCA(n_components=2).fit(X)
    b = pca.PCA(n_components=2).fit(X + 100.0)

    # Components are unique up to sign; compare absolute dot products.
    cos = np.abs(np.diag(a.components_ @ b.components_.T))
    np.testing.assert_allclose(cos, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Practical use: classification quality after dimensionality reduction
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="PCA not implemented", strict=False)
def test_iris_first_two_components_separate_classes(iris):
    """A k-NN trained on the top-2 PCs should reach >= 90% accuracy on Iris.

    This catches sign / scaling bugs that don't trip the unit-tests above
    but would still ruin a downstream classifier.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    X, y = iris
    model = pca.PCA(n_components=2)
    Z = model.fit(X).transform(X)
    score = cross_val_score(KNeighborsClassifier(), Z, y, cv=5).mean()
    assert score >= 0.90
