"""Tests for the from-scratch transformer.

Module under test: ``src.supervised.neural_networks.transformer`` (path TBD).

Datasets
--------
* Tiny Shakespeare — in-memory string in ``conftest.py``. Used for fast
  character-level smoke tests (training loop, attention shapes, loss
  decreases). No network or disk I/O required.
* WikiText-2       — language-modeling benchmark loaded via HuggingFace
  ``datasets``. Skipped when offline.
* IMDB             — sentiment classification benchmark. Used to test the
  model in a classification head configuration.

What these tests should cover once the model is implemented
-----------------------------------------------------------
1. Component shapes:
   - MultiHeadSelfAttention: input (B, T, D) -> output (B, T, D), and the
     attention-weight tensor is (B, H, T, T) with rows summing to 1.
   - Causal mask: future positions receive ~0 attention weight.
2. Positional encoding:
   - Sinusoidal encoding for two adjacent positions differs by a small
     known delta; encodings repeat with the expected period.
3. Forward pass:
   - Logits shape is (B, T, vocab_size) for LM, or (B, n_classes) for
     classification head.
4. Training:
   - On Tiny Shakespeare, validation cross-entropy after 200 steps is below
     the unigram baseline (log of vocab size). This is intentionally a low
     bar - only verifies the model is *learning*, not that it's good.
   - On a 2k-sample IMDB subset, classification accuracy >= 70% after 3
     epochs (IMDB is easy; this catches gross bugs).
5. Determinism:
   - Same seed -> identical first-step loss.
"""

from __future__ import annotations

import math

import pytest

transformer = pytest.importorskip(
    "src.supervised.neural_networks.transformer"
)


# ---------------------------------------------------------------------------
# Component-level shape and mask tests
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="transformer not implemented", strict=False)
def test_self_attention_shapes():
    """MHSA preserves (B, T, D) and yields (B, H, T, T) attention weights."""
    import torch

    B, T, D, H = 2, 7, 16, 4
    block = transformer.MultiHeadSelfAttention(d_model=D, n_heads=H)
    x = torch.randn(B, T, D)
    out, attn = block(x, return_attention=True)
    assert out.shape == (B, T, D)
    assert attn.shape == (B, H, T, T)
    # Each row of the attention matrix is a probability distribution.
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, H, T), atol=1e-6)


@pytest.mark.xfail(reason="transformer not implemented", strict=False)
def test_causal_mask_blocks_future_tokens():
    """With a causal mask, attention weight to any future position must be ~0."""
    import torch

    B, T, D, H = 1, 5, 16, 2
    block = transformer.MultiHeadSelfAttention(d_model=D, n_heads=H, causal=True)
    x = torch.randn(B, T, D)
    _, attn = block(x, return_attention=True)
    # Upper-triangular (excluding diagonal) entries should be ~0.
    upper = torch.triu(attn[0, 0], diagonal=1)
    assert torch.all(upper.abs() < 1e-6)


@pytest.mark.xfail(reason="transformer not implemented", strict=False)
def test_positional_encoding_periodicity():
    """Sinusoidal positional encodings differ between adjacent positions."""
    import torch

    pe = transformer.SinusoidalPositionalEncoding(d_model=32, max_len=128)
    enc = pe(torch.zeros(1, 128, 32))
    assert not torch.allclose(enc[0, 0], enc[0, 1])


# ---------------------------------------------------------------------------
# End-to-end LM smoke test on Tiny Shakespeare
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="transformer not implemented", strict=False)
def test_tiny_shakespeare_loss_decreases(tiny_shakespeare, random_seed):
    """A tiny LM should train below unigram entropy on Tiny Shakespeare.

    Char-level vocab is small (~50), so log(vocab) ~= 3.9. Achievable target
    after a handful of steps without any optimization tricks.
    """
    text = tiny_shakespeare
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = [stoi[c] for c in text]

    model = transformer.TinyLM(
        vocab_size=len(vocab),
        d_model=32,
        n_heads=2,
        n_layers=2,
        block_size=64,
        random_state=random_seed,
    )
    final_loss = model.fit_lm(ids, n_steps=200, batch_size=16, lr=3e-3)
    assert final_loss < math.log(len(vocab))


# ---------------------------------------------------------------------------
# Classification head on IMDB
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.xfail(reason="transformer not implemented", strict=False)
def test_imdb_classification_head(imdb_dataset, random_seed):
    """Transformer-with-classifier-head reaches >= 70% accuracy on IMDB subset."""
    train = imdb_dataset["train"].shuffle(seed=random_seed).select(range(2_000))
    test = imdb_dataset["test"].shuffle(seed=random_seed).select(range(500))

    model = transformer.TinyClassifier(
        vocab_size=8_192,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_classes=2,
        random_state=random_seed,
    )
    model.fit_classifier(
        texts=train["text"], labels=train["label"], n_epochs=3, batch_size=32
    )
    preds = model.predict(test["text"])
    acc = (preds == test["label"]).mean()
    assert acc >= 0.70


# ---------------------------------------------------------------------------
# Optional WikiText-2 perplexity check
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.xfail(reason="transformer not implemented", strict=False)
def test_wikitext2_perplexity_below_baseline(wikitext2_dataset, random_seed):
    """Validation perplexity should be lower than uniform-over-vocab baseline."""
    text = "\n".join(wikitext2_dataset["validation"]["text"][:500])
    # Tokenization is up to the implementation - keep this assertion loose.
    model = transformer.TinyLM(
        vocab_size=4_096, d_model=64, n_heads=4, n_layers=2, block_size=128,
        random_state=random_seed,
    )
    ppl = model.eval_perplexity(text)
    assert ppl < 4_096
