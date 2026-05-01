"""Tests for the from-scratch GAN.

Module under test: ``src.generative.gan`` (path TBD).

Datasets
--------
* FashionMNIST — primary GAN benchmark. 28x28 grayscale, easier than CelebA
  but rich enough that mode collapse / dead generator bugs become visible.
* CelebA       — optional advanced dataset. Tests using it are gated on the
  data being cached locally and are marked ``slow``.

What these tests should cover once the model is implemented
-----------------------------------------------------------
GANs are notoriously hard to test - we cannot assert "the samples look like
clothing" - so we lean on *structural* and *behavioral* properties:

1. Generator API:
   - Accepts a (B, latent_dim) noise tensor and returns images of the
     expected shape ((B, 1, 28, 28) for FashionMNIST).
   - Output values are in the documented range (e.g. [-1, 1] for tanh).
2. Discriminator API:
   - Accepts (B, C, H, W) images and returns (B,) or (B, 1) logits/probs.
3. Training step:
   - Each call to ``train_step`` runs both D and G updates exactly once and
     returns finite, non-NaN losses.
   - After N steps on FashionMNIST, the discriminator's accuracy on real vs.
     fake samples is in (0.5, 0.95) - i.e. it learned something but the
     generator hasn't completely collapsed.
4. Reproducibility:
   - Same seed -> identical first-batch generator output.
5. Sample diversity:
   - Pairwise distance between 64 generated samples has nonzero variance
     (catches mode collapse where every sample is identical).
"""

from __future__ import annotations

from pathlib import Path

import pytest

gan = pytest.importorskip("src.generative.gan")


# ---------------------------------------------------------------------------
# Generator / Discriminator shape contracts
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="GAN not implemented", strict=False)
def test_generator_output_shape_and_range(random_seed):
    """G should map noise to images in [-1, 1] with shape (B, 1, 28, 28)."""
    import torch

    torch.manual_seed(random_seed)
    G = gan.Generator(latent_dim=64, image_channels=1, image_size=28)
    z = torch.randn(8, 64)
    fake = G(z)
    assert fake.shape == (8, 1, 28, 28)
    assert fake.min() >= -1.0 - 1e-6
    assert fake.max() <= 1.0 + 1e-6


@pytest.mark.xfail(reason="GAN not implemented", strict=False)
def test_discriminator_output_shape():
    """D should map images to a (B,) or (B, 1) score."""
    import torch

    D = gan.Discriminator(image_channels=1, image_size=28)
    x = torch.randn(8, 1, 28, 28)
    score = D(x)
    assert score.shape in {(8,), (8, 1)}


# ---------------------------------------------------------------------------
# Training step on FashionMNIST
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.xfail(reason="GAN not implemented", strict=False)
def test_train_step_returns_finite_losses(fashion_mnist_train, random_seed):
    """A single train_step on a real batch should produce finite D and G losses."""
    import math

    import torch

    loader = torch.utils.data.DataLoader(
        fashion_mnist_train, batch_size=64, shuffle=True
    )
    real, _ = next(iter(loader))

    trainer = gan.GANTrainer(latent_dim=64, image_size=28, random_state=random_seed)
    losses = trainer.train_step(real)
    assert math.isfinite(losses["d_loss"])
    assert math.isfinite(losses["g_loss"])


@pytest.mark.slow
@pytest.mark.xfail(reason="GAN not implemented", strict=False)
def test_discriminator_does_not_collapse(fashion_mnist_train, random_seed):
    """After short training the discriminator should be in the 'learning' band.

    If D acc >= 0.95 the generator collapsed; if D acc <= 0.55 D never learned.
    Either is a red flag. The (0.55, 0.95) window is generous so real GANs
    pass even with their typical instability.
    """
    import torch

    loader = torch.utils.data.DataLoader(
        fashion_mnist_train, batch_size=64, shuffle=True
    )
    trainer = gan.GANTrainer(latent_dim=64, image_size=28, random_state=random_seed)
    for i, (real, _) in enumerate(loader):
        trainer.train_step(real)
        if i >= 50:  # ~50 D/G updates, plenty for a smoke test
            break

    real, _ = next(iter(loader))
    d_acc = trainer.discriminator_accuracy(real, n_fake=real.size(0))
    assert 0.55 < d_acc < 0.95


# ---------------------------------------------------------------------------
# Reproducibility and diversity
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="GAN not implemented", strict=False)
def test_same_seed_same_initial_samples(random_seed):
    """Two trainers with the same seed must produce identical first samples."""
    import torch

    a = gan.GANTrainer(latent_dim=32, image_size=28, random_state=random_seed)
    b = gan.GANTrainer(latent_dim=32, image_size=28, random_state=random_seed)
    z = torch.randn(4, 32)
    torch.testing.assert_close(a.generator(z), b.generator(z))


@pytest.mark.xfail(reason="GAN not implemented", strict=False)
def test_samples_are_diverse(random_seed):
    """64 generated samples should not all collapse to the same image.

    Mode collapse manifests as near-zero pairwise distance variance.
    """
    import torch

    G = gan.Generator(latent_dim=64, image_channels=1, image_size=28)
    z = torch.randn(64, 64)
    samples = G(z).flatten(start_dim=1)
    pdist = torch.cdist(samples, samples)
    # Use upper-triangle (i < j) entries to avoid the zero diagonal.
    iu = torch.triu_indices(64, 64, offset=1)
    distances = pdist[iu[0], iu[1]]
    assert distances.var().item() > 1e-4


# ---------------------------------------------------------------------------
# Optional: CelebA
# ---------------------------------------------------------------------------


CELEBA_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "celeba"


@pytest.mark.slow
@pytest.mark.skipif(
    not CELEBA_ROOT.exists(),
    reason="Optional dataset; place CelebA under data/raw/celeba to enable.",
)
@pytest.mark.xfail(reason="GAN not implemented", strict=False)
def test_celeba_generator_runs(random_seed):
    """End-to-end smoke test on CelebA: one train step on 64x64 RGB images."""
    import torch
    from torchvision import datasets, transforms

    tx = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64),
                             transforms.ToTensor()])
    dataset = datasets.ImageFolder(str(CELEBA_ROOT), transform=tx)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    real, _ = next(iter(loader))

    trainer = gan.GANTrainer(
        latent_dim=128, image_channels=3, image_size=64, random_state=random_seed
    )
    losses = trainer.train_step(real)
    assert all(map(lambda v: v == v, losses.values()))  # NaN check
