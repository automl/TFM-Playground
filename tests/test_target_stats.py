"""Degenerate-case tests for target-stat computation in `normalization.py`.

These pin the behavior when the training targets don't admit a meaningful
spread: a single training sample (unbiased std is undefined -> NaN) or
constant targets (std == 0). In both cases the standardization must stay
finite, so the std is forced to 1.0. Normal inputs must be left untouched.
"""

import numpy as np
import torch

from tfmplayground.normalization import (
    TARGET_NORM_EPS,
    compute_target_stats_numpy,
    compute_target_stats_torch,
)


# --- numpy path (inference, single dataset) ---------------------------------


def test_numpy_single_sample_std_is_finite_and_one():
    """One training target: unbiased std is NaN, which today poisons every
    prediction. The guard must return a finite std of 1.0 instead.
    """
    mean, std = compute_target_stats_numpy(np.array([5.0]))

    assert np.isfinite(mean) and mean == 5.0
    assert np.isfinite(std)
    assert std == 1.0


def test_numpy_constant_targets_std_is_one():
    """Constant targets have std 0; dividing by ~0 blows up the scale.
    The guard replaces it with 1.0 (not the tiny 1e-8 epsilon).
    """
    _, std = compute_target_stats_numpy(np.array([3.0, 3.0, 3.0]))

    assert std == 1.0


def test_numpy_normal_case_is_unchanged():
    """Regression guard: for a normal spread of targets the result must be
    exactly what it is today (raw unbiased std + epsilon).
    """
    y = np.array([0.0, 2.0, 4.0])
    mean, std = compute_target_stats_numpy(y)

    assert mean == np.mean(y)
    assert std == np.std(y, ddof=1) + TARGET_NORM_EPS


# --- torch path (training, batched) -----------------------------------------


def test_torch_single_datapoint_std_is_finite_and_one():
    """Batch of one dataset with a single datapoint: std over dim=1 is NaN
    today. The guard must return a finite std of 1.0.
    """
    y = torch.tensor([[[5.0]]])  # (batch=1, n=1, 1)
    _, std = compute_target_stats_torch(y)

    assert torch.isfinite(std).all()
    assert torch.allclose(std, torch.ones_like(std))


def test_torch_constant_targets_std_is_one():
    """Constant targets -> std 0 today (saved only by 1e-8). Guard -> 1.0."""
    y = torch.tensor([[[3.0], [3.0], [3.0]]])  # (1, 3, 1)
    _, std = compute_target_stats_torch(y)

    assert torch.allclose(std, torch.ones_like(std))


def test_torch_only_degenerate_datasets_in_batch_are_fixed():
    """The std is per-dataset within the batch, so the guard must be applied
    elementwise: a normal dataset keeps its real std while a degenerate one
    (here constant) is forced to 1.0.
    """
    normal = torch.tensor([[0.0], [2.0], [4.0]])  # std (ddof=1) == 2.0
    constant = torch.tensor([[7.0], [7.0], [7.0]])  # std == 0
    y = torch.stack([normal, constant], dim=0)  # (2, 3, 1)

    _, std = compute_target_stats_torch(y)

    assert torch.isfinite(std).all()
    assert torch.allclose(std[0], torch.full_like(std[0], 2.0), atol=1e-6)
    assert torch.allclose(std[1], torch.ones_like(std[1]))


def test_torch_normal_case_is_unchanged():
    """Regression guard for the batched path: normal spread -> raw std + eps."""
    y = torch.tensor([[[0.0], [2.0], [4.0]]])  # (1, 3, 1)
    mean, std = compute_target_stats_torch(y)

    expected_std = y.std(dim=1, keepdim=True) + TARGET_NORM_EPS
    assert torch.allclose(std, expected_std)
    assert torch.allclose(mean, y.mean(dim=1, keepdim=True))