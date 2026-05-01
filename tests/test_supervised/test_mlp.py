"""Tests for the from-scratch multi-layer perceptron (MLP).

Module under test: ``src.supervised.neural_networks.mlp`` (path TBD).

Datasets
--------
* MNIST          — 60k train / 10k test, 28x28 grayscale digits, 10 classes.
* FashionMNIST   — same shape as MNIST but harder. Used to exercise the same
  architecture on a non-trivially overlapping class distribution.

Both are loaded via torchvision (see ``conftest.py``); tests are skipped when
the cached data is unavailable. To enable downloading on first run set
``MLFS_ALLOW_DOWNLOAD=1`` in the environment.

What these tests should cover once the model is implemented
-----------------------------------------------------------
1. Forward pass:
   - Output shape (batch, n_classes).
   - Softmax probabilities sum to 1 per row.
2. Backward pass:
   - Numeric gradient check on a tiny network (e.g. 4-3-2) matches analytic
     gradients to ~1e-5 relative error.
3. Training:
   - On a 5k-sample MNIST subset, a 2-hidden-layer MLP reaches >= 90% test
     accuracy in <= 5 epochs. Tight enough to catch regressions, loose enough
     to remain fast in CI (target ~30s on CPU).
   - On a similar FashionMNIST subset, accuracy reaches >= 80%.
4. Reproducibility:
   - Same seed yields identical loss curves.
"""

from __future__ import annotations

import numpy as np
import pytest

mlp = pytest.importorskip("src.supervised.neural_networks.mlp")


# ---------------------------------------------------------------------------
# Numeric gradient check on a tiny problem
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="MLP not implemented", strict=False)
def test_gradient_check_tiny_network(random_seed):
    """Compare analytic backprop gradients to centered-difference numeric gradients.

    Done on a tiny network so the O(n_params) numeric gradient is fast.
    Relative error <= 1e-5 is the standard textbook threshold (Bishop 5.3.4).
    """
    rng = np.random.default_rng(random_seed)
    X = rng.standard_normal((8, 4))
    y = rng.integers(0, 2, size=8)

    model = mlp.MLPClassifier(hidden_sizes=(3,), random_state=random_seed)
    model.fit(X, y, n_epochs=0)  # init weights without training

    analytic = model.compute_gradients(X, y)
    numeric = model.numerical_gradients(X, y, eps=1e-5)
    for name, g_an in analytic.items():
        g_num = numeric[name]
        rel_err = np.abs(g_an - g_num) / (np.abs(g_an) + np.abs(g_num) + 1e-12)
        assert rel_err.max() < 1e-5, f"gradient mismatch in {name}"


# ---------------------------------------------------------------------------
# MNIST end-to-end
# ---------------------------------------------------------------------------


def _flatten_subset(dataset, n: int):
    """Return the first ``n`` items of an image dataset as flat numpy arrays.

    Defined locally so we don't depend on the implementation's data API.
    """
    import torch

    loader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=False)
    images, labels = next(iter(loader))
    return images.view(n, -1).numpy(), labels.numpy()


@pytest.mark.slow
@pytest.mark.xfail(reason="MLP not implemented", strict=False)
def test_mnist_subset_accuracy(mnist_train, mnist_test, random_seed):
    """A small MNIST subset should reach >= 90% test accuracy in a few epochs."""
    X_train, y_train = _flatten_subset(mnist_train, 5_000)
    X_test, y_test = _flatten_subset(mnist_test, 1_000)

    model = mlp.MLPClassifier(
        hidden_sizes=(128, 64),
        learning_rate=1e-3,
        n_epochs=5,
        batch_size=64,
        random_state=random_seed,
    )
    model.fit(X_train, y_train)
    acc = (model.predict(X_test) == y_test).mean()
    assert acc >= 0.90, f"MNIST accuracy too low: {acc:.3f}"


@pytest.mark.slow
@pytest.mark.xfail(reason="MLP not implemented", strict=False)
def test_fashion_mnist_subset_accuracy(
    fashion_mnist_train, fashion_mnist_test, random_seed
):
    """FashionMNIST should reach >= 80% test accuracy on the same recipe."""
    X_train, y_train = _flatten_subset(fashion_mnist_train, 5_000)
    X_test, y_test = _flatten_subset(fashion_mnist_test, 1_000)

    model = mlp.MLPClassifier(
        hidden_sizes=(128, 64),
        learning_rate=1e-3,
        n_epochs=5,
        batch_size=64,
        random_state=random_seed,
    )
    model.fit(X_train, y_train)
    acc = (model.predict(X_test) == y_test).mean()
    assert acc >= 0.80, f"FashionMNIST accuracy too low: {acc:.3f}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="MLP not implemented", strict=False)
def test_same_seed_same_loss_curve(random_seed):
    """Two fits with the same seed must produce bit-identical loss histories."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((128, 10))
    y = rng.integers(0, 3, size=128)

    a = mlp.MLPClassifier(hidden_sizes=(8,), random_state=random_seed, n_epochs=3)
    b = mlp.MLPClassifier(hidden_sizes=(8,), random_state=random_seed, n_epochs=3)
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_array_equal(a.loss_history_, b.loss_history_)
