import pytest
import torch

from tfmplayground.augmentation import add_widening_features


def test_widening_shape_without_originals():
    """Without appending originals, output has exactly num_features_to_add features."""
    torch.manual_seed(0)
    x = torch.randn(2, 30, 4)
    out = add_widening_features(x, num_features_to_add=10, sparsity=0.05, noise_std=0.5, include_original_prob=0.0)
    assert out.shape == (2, 30, 10)


def test_widening_shape_with_originals():
    """Appending originals gives num_original + num_features_to_add features."""
    torch.manual_seed(0)
    x = torch.randn(2, 30, 4)
    out = add_widening_features(x, num_features_to_add=10, sparsity=0.05, noise_std=0.5, include_original_prob=1.0)
    assert out.shape == (2, 30, 14)


def test_widening_sparsity_zero_noise_zero_is_all_zero():
    """sparsity=0 zeroes the projection and noise_std=0 adds nothing, so the new block is 0."""
    torch.manual_seed(0)
    x = torch.randn(2, 30, 4)
    out = add_widening_features(x, num_features_to_add=6, sparsity=0.0, noise_std=0.0, include_original_prob=0.0)
    assert torch.all(out == 0.0)


def test_widening_does_not_mutate_input():
    """The input tensor is never modified in place."""
    torch.manual_seed(0)
    x = torch.randn(2, 30, 4)
    x_before = x.clone()
    add_widening_features(x, num_features_to_add=5, sparsity=0.1, noise_std=0.3)
    assert torch.equal(x, x_before)


def test_widening_is_reproducible_with_generator():
    """The same generator seed produces identical widened features."""
    x = torch.randn(2, 30, 4)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    out1 = add_widening_features(x, 8, sparsity=0.1, noise_std=0.3, generator=g1)
    out2 = add_widening_features(x, 8, sparsity=0.1, noise_std=0.3, generator=g2)
    assert torch.equal(out1, out2)


def test_widening_rejects_invalid_sparsity():
    """A sparsity outside [0, 1] is an error."""
    x = torch.randn(2, 30, 4)
    with pytest.raises(ValueError):
        add_widening_features(x, 5, sparsity=1.5, noise_std=0.1)


def test_widening_all_zero_projection_becomes_pure_noise():
    """With no kept weights the projection is 0; like the reference implementation, the
    noise must still be added (std treated as 1), giving pure-noise features, not constants."""
    torch.manual_seed(0)
    x = torch.randn(2, 30, 4)
    out = add_widening_features(x, num_features_to_add=6, sparsity=0.0, noise_std=1.0, include_original_prob=0.0)
    assert torch.all(out.std(dim=-2) > 0)