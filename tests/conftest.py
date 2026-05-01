"""Shared pytest fixtures for the ml-from-scratch test suite.

Fixtures defined here are available to every test module without explicit import.
They centralize dataset loading so that individual model tests stay focused on
behavior assertions rather than I/O plumbing.

Dataset sourcing policy
-----------------------
* In-memory sklearn datasets (Iris, Wine, California Housing, digits, blobs)
  are always loaded; they are tiny and ship with scikit-learn.
* Image datasets (MNIST, FashionMNIST, CIFAR10) are loaded via torchvision.
  Tests that depend on them are skipped when torchvision or its cached data
  is unavailable (e.g. offline CI without a pre-populated cache).
* Text datasets (IMDB, WikiText-2, Shakespeare) are loaded via HuggingFace
  `datasets` or a tiny in-repo file. Tests that depend on them are skipped
  when `datasets` cannot reach the internet.

Determinism
-----------
A `random_seed` fixture exposes the canonical seed used everywhere so that
stochastic models (MLP/CNN/GAN/Transformer init, K-Means seeding, GBM column
subsampling) produce reproducible numbers across runs and machines.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# Canonical seed for any test that needs reproducible randomness.
RANDOM_SEED = 1337

# Where torchvision should look for / cache image datasets. Using a project-
# local path keeps CI runs self-contained and avoids polluting $HOME.
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"


# ---------------------------------------------------------------------------
# Generic fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def random_seed() -> int:
    """Return the global random seed used across the test suite."""
    return RANDOM_SEED


@pytest.fixture(autouse=True)
def _seed_numpy(random_seed: int) -> None:
    """Reseed NumPy's legacy global RNG before every test for reproducibility.

    Autouse so individual tests don't have to remember to seed. Tests that
    need a torch RNG should additionally call `torch.manual_seed` themselves
    to keep the torch dependency optional at collection time.
    """
    np.random.seed(random_seed)


# ---------------------------------------------------------------------------
# sklearn tabular datasets
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def california_housing():
    """Regression dataset: 20,640 samples × 8 features predicting median house value.

    Used by linear regression and gradient boosting regressor tests.
    """
    from sklearn.datasets import fetch_california_housing

    bunch = fetch_california_housing()
    return bunch.data, bunch.target


@pytest.fixture(scope="session")
def iris():
    """Classification dataset: 150 samples × 4 features, 3 balanced classes.

    Used by logistic regression, K-Means, and PCA tests.
    """
    from sklearn.datasets import load_iris

    bunch = load_iris()
    return bunch.data, bunch.target


@pytest.fixture(scope="session")
def wine():
    """Classification dataset: 178 samples × 13 features, 3 classes.

    Used by logistic regression and gradient boosting classifier tests.
    """
    from sklearn.datasets import load_wine

    bunch = load_wine()
    return bunch.data, bunch.target


@pytest.fixture(scope="session")
def digits():
    """Classification dataset: 1,797 8x8 grayscale digit images flattened to 64 features.

    Used by PCA (dimensionality reduction sanity check) and K-Means tests.
    """
    from sklearn.datasets import load_digits

    bunch = load_digits()
    return bunch.data, bunch.target


@pytest.fixture(scope="session")
def blobs(random_seed: int):
    """Synthetic clustering dataset with 4 well-separated isotropic Gaussian blobs.

    Returns (X, y) where y is the true cluster label. Used by K-Means tests
    where ARI/NMI against ground truth should be near 1.0 if the algorithm
    is implemented correctly.
    """
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        n_samples=500,
        centers=4,
        cluster_std=0.8,
        n_features=2,
        random_state=random_seed,
    )
    return X, y


# ---------------------------------------------------------------------------
# Image datasets (torchvision)
# ---------------------------------------------------------------------------


def _load_torchvision_dataset(name: str, train: bool):
    """Internal helper: load a torchvision dataset, skip the test if unavailable."""
    torchvision = pytest.importorskip("torchvision")
    from torchvision import transforms

    # Channel-first float tensor in [0, 1]; tests can override via their own transforms.
    transform = transforms.ToTensor()

    dataset_cls = getattr(torchvision.datasets, name)
    try:
        # download=False so offline CI fails fast and the test is skipped rather
        # than hanging on a network call.
        return dataset_cls(
            root=str(DATA_ROOT),
            train=train,
            download=os.environ.get("MLFS_ALLOW_DOWNLOAD") == "1",
            transform=transform,
        )
    except RuntimeError as exc:  # raised by torchvision when files are missing
        pytest.skip(f"{name} dataset not cached locally: {exc}")


@pytest.fixture(scope="session")
def mnist_train():
    """MNIST training split (60k 28x28 grayscale digits). Used by MLP and CNN tests."""
    return _load_torchvision_dataset("MNIST", train=True)


@pytest.fixture(scope="session")
def mnist_test():
    """MNIST test split (10k samples)."""
    return _load_torchvision_dataset("MNIST", train=False)


@pytest.fixture(scope="session")
def fashion_mnist_train():
    """FashionMNIST training split (60k 28x28 grayscale apparel images).

    Used by MLP, CNN, and GAN tests; harder than MNIST so accuracy thresholds
    in tests should be lower.
    """
    return _load_torchvision_dataset("FashionMNIST", train=True)


@pytest.fixture(scope="session")
def fashion_mnist_test():
    """FashionMNIST test split (10k samples)."""
    return _load_torchvision_dataset("FashionMNIST", train=False)


@pytest.fixture(scope="session")
def cifar10_train():
    """CIFAR-10 training split (50k 32x32 RGB images, 10 classes).

    Optional dataset for CNN tests that want a harder benchmark than MNIST.
    """
    return _load_torchvision_dataset("CIFAR10", train=True)


@pytest.fixture(scope="session")
def cifar10_test():
    """CIFAR-10 test split (10k samples)."""
    return _load_torchvision_dataset("CIFAR10", train=False)


# ---------------------------------------------------------------------------
# Text datasets
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_shakespeare():
    """Tiny in-memory Shakespeare corpus for fast transformer smoke tests.

    Returns a single string of ~1KB - just enough text to verify tokenization,
    forward/backward passes, and that the loss decreases over a few steps.
    Larger corpora (WikiText-2, IMDB) are loaded via separate fixtures and
    used in slower / optional tests.
    """
    return (
        "To be, or not to be, that is the question:\n"
        "Whether 'tis nobler in the mind to suffer\n"
        "The slings and arrows of outrageous fortune,\n"
        "Or to take arms against a sea of troubles\n"
        "And by opposing end them. To die-to sleep,\n"
        "No more; and by a sleep to say we end\n"
        "The heart-ache and the thousand natural shocks\n"
        "That flesh is heir to: 'tis a consummation\n"
        "Devoutly to be wish'd. To die, to sleep;\n"
        "To sleep, perchance to dream-ay, there's the rub:\n"
        "For in that sleep of death what dreams may come,\n"
        "When we have shuffled off this mortal coil,\n"
        "Must give us pause-there's the respect\n"
        "That makes calamity of so long life.\n"
    ) * 4


@pytest.fixture(scope="session")
def imdb_dataset():
    """HuggingFace IMDB dataset for sentiment classification (transformer tests).

    Skipped when `datasets` is missing or offline. Returns the full DatasetDict
    so tests can pick the splits and slice sizes they need.
    """
    datasets = pytest.importorskip("datasets")
    try:
        return datasets.load_dataset("imdb")
    except Exception as exc:  # connection / hub errors
        pytest.skip(f"IMDB dataset unavailable: {exc}")


@pytest.fixture(scope="session")
def wikitext2_dataset():
    """HuggingFace WikiText-2 (raw v1) for language-modeling transformer tests.

    Skipped when `datasets` is missing or offline.
    """
    datasets = pytest.importorskip("datasets")
    try:
        return datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
    except Exception as exc:
        pytest.skip(f"WikiText-2 dataset unavailable: {exc}")
