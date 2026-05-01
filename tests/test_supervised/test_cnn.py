"""Tests for the from-scratch convolutional neural network.

Module under test: ``src.supervised.neural_networks.cnn`` (path TBD).

Datasets
--------
* MNIST          — easy baseline; CNN should comfortably exceed the MLP score.
* FashionMNIST   — primary CNN benchmark; harder than MNIST, still small.
* CIFAR-10       — optional, harder benchmark (32x32 RGB). Tests using it are
  marked ``slow`` and skipped unless CIFAR-10 is cached locally.

What these tests should cover once the model is implemented
-----------------------------------------------------------
1. Layer correctness:
   - Conv2d output shape obeys (H + 2p - k)/s + 1.
   - MaxPool2d halves spatial dims with kernel=2 stride=2.
   - Flatten + Linear closes the model into class logits.
2. Forward / backward:
   - Logits shape is (batch, n_classes); cross-entropy reduces to a scalar.
   - Numeric gradient check on a 2-conv-layer toy net matches analytic grads
     to <= 1e-4 relative error (looser than MLP because of conv stencil).
3. Training:
   - MNIST subset reaches >= 95% test accuracy in 3 epochs.
   - FashionMNIST subset reaches >= 85% test accuracy in 3 epochs.
   - CIFAR-10 (optional) reaches >= 55% test accuracy in 5 epochs - intentionally
     a low bar; the goal is to verify the implementation can train on RGB at all.
4. Determinism:
   - Same seed produces identical first-epoch loss.
"""

from __future__ import annotations

import pytest

cnn = pytest.importorskip("src.supervised.neural_networks.cnn")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_subset(dataset, n: int):
    """Return the first ``n`` items of an image dataset as (NCHW tensor, labels)."""
    import torch

    loader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=False)
    images, labels = next(iter(loader))
    return images, labels


# ---------------------------------------------------------------------------
# Layer-shape sanity checks
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="CNN not implemented", strict=False)
def test_conv_output_shape():
    """Conv2d with kernel=3, stride=1, padding=1 preserves spatial dims."""
    import torch

    layer = cnn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
    x = torch.zeros(2, 1, 28, 28)
    assert layer(x).shape == (2, 8, 28, 28)


@pytest.mark.xfail(reason="CNN not implemented", strict=False)
def test_pool_output_shape():
    """MaxPool2d kernel=2 stride=2 halves H and W."""
    import torch

    layer = cnn.MaxPool2d(kernel_size=2, stride=2)
    assert layer(torch.zeros(2, 8, 28, 28)).shape == (2, 8, 14, 14)


# ---------------------------------------------------------------------------
# End-to-end training smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.xfail(reason="CNN not implemented", strict=False)
def test_mnist_cnn_accuracy(mnist_train, mnist_test, random_seed):
    """CNN should reach >= 95% test accuracy on a 5k-sample MNIST subset."""
    X_train, y_train = _image_subset(mnist_train, 5_000)
    X_test, y_test = _image_subset(mnist_test, 1_000)

    model = cnn.SmallCNN(num_classes=10, random_state=random_seed)
    model.fit(X_train, y_train, n_epochs=3, batch_size=64)
    acc = (model.predict(X_test) == y_test).float().mean().item()
    assert acc >= 0.95


@pytest.mark.slow
@pytest.mark.xfail(reason="CNN not implemented", strict=False)
def test_fashion_mnist_cnn_accuracy(
    fashion_mnist_train, fashion_mnist_test, random_seed
):
    """FashionMNIST is harder; require >= 85%."""
    X_train, y_train = _image_subset(fashion_mnist_train, 5_000)
    X_test, y_test = _image_subset(fashion_mnist_test, 1_000)

    model = cnn.SmallCNN(num_classes=10, random_state=random_seed)
    model.fit(X_train, y_train, n_epochs=3, batch_size=64)
    acc = (model.predict(X_test) == y_test).float().mean().item()
    assert acc >= 0.85


@pytest.mark.slow
@pytest.mark.xfail(reason="CNN not implemented", strict=False)
def test_cifar10_cnn_smoke(cifar10_train, cifar10_test, random_seed):
    """CIFAR-10 sanity check: just confirm the model can train above chance."""
    X_train, y_train = _image_subset(cifar10_train, 5_000)
    X_test, y_test = _image_subset(cifar10_test, 1_000)

    model = cnn.SmallCNN(num_classes=10, in_channels=3, random_state=random_seed)
    model.fit(X_train, y_train, n_epochs=5, batch_size=64)
    acc = (model.predict(X_test) == y_test).float().mean().item()
    assert acc >= 0.55
