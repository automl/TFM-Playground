"""Single source of truth for regression target normalization.

Training (torch, batched, gradient-tracked) and inference (numpy, single
dataset) must standardize targets identically, otherwise the model would be
trained and queried on different target scales. Keeping the spec here means
the two paths cannot silently drift apart.

Spec: standardize by mean and unbiased standard deviation (ddof=1), with a
small epsilon added to the std for numerical stability.
"""

import numpy as np
import torch

# Added to the standard deviation to avoid division by (near-)zero.
TARGET_NORM_EPS = 1e-8


def compute_target_stats_torch(y_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std of batched targets over the sample dimension (dim=1).

    Uses unbiased std (ddof=1) and keeps gradients, so it is safe to call
    inside the training loop.

    Args:
        y_train: tensor of shape (batch_size, num_train_datapoints, 1)

    Returns:
        (mean, std), each of shape (batch_size, 1, 1), broadcastable over y.
    """
    mean = y_train.mean(dim=1, keepdim=True)
    std = y_train.std(dim=1, keepdim=True) + TARGET_NORM_EPS
    return mean, std


def compute_target_stats_numpy(y_train: np.ndarray) -> tuple[np.floating, np.floating]:
    """Mean and std of a single target array.

    Uses unbiased std (ddof=1) to match compute_target_stats_torch.

    Args:
        y_train: 1D array of training targets.

    Returns:
        (mean, std) as scalars.
    """
    mean = np.mean(y_train)
    std = np.std(y_train, ddof=1) + TARGET_NORM_EPS
    return mean, std


def normalize_targets(y, mean, std):
    """Standardize targets: (y - mean) / std. Works for torch and numpy."""
    return (y - mean) / std


def denormalize_predictions(preds, mean, std):
    """Invert normalize_targets: preds * std + mean. Works for torch and numpy."""
    return preds * std + mean