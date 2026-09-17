"""Single source of truth for regression target normalization.

Training (torch, batched, gradient-tracked) and inference (numpy, single
dataset) must standardize targets identically, otherwise the model would be
trained and queried on different target scales. Keeping the spec here means
the two paths cannot silently drift apart.

Spec: standardize by mean and unbiased standard deviation (ddof=1), with a
small epsilon added to the std for numerical stability. Degenerate inputs
whose std is undefined or zero -- a single training sample (ddof=1 std is
NaN) or constant targets (std is 0) -- use a std of 1.0 so the
standardization stays finite instead of producing NaNs or exploding.
"""

import warnings

import numpy as np
import torch

# Added to the standard deviation to avoid division by (near-)zero.
TARGET_NORM_EPS = 1e-8


def compute_target_stats_torch(y_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std of batched targets over the sample dimension (dim=1).

    Uses unbiased std (ddof=1) and keeps gradients, so it is safe to call
    inside the training loop. Datasets in the batch whose std is non-finite
    or zero (single datapoint or constant target) get a std of 1.0.

    Args:
        y_train: tensor of shape (batch_size, num_train_datapoints, 1)

    Returns:
        (mean, std), each of shape (batch_size, 1, 1), broadcastable over y.
    """
    mean = y_train.mean(dim=1, keepdim=True)
    with warnings.catch_warnings():
        # A single datapoint makes unbiased std warn; we handle it below.
        warnings.simplefilter("ignore")
        std = y_train.std(dim=1, keepdim=True)
    # Guard degenerate datasets elementwise, since std is computed per dataset.
    degenerate = ~torch.isfinite(std) | (std == 0)
    std = torch.where(degenerate, torch.ones_like(std), std + TARGET_NORM_EPS)
    return mean, std


def compute_target_stats_numpy(y_train: np.ndarray) -> tuple[np.floating, np.floating]:
    """Mean and std of a single target array.

    Uses unbiased std (ddof=1) to match compute_target_stats_torch. A single
    sample (std is NaN) or constant targets (std is 0) get a std of 1.0.

    Args:
        y_train: 1D array of training targets.

    Returns:
        (mean, std) as scalars.
    """
    mean = np.mean(y_train)
    with warnings.catch_warnings():
        # A single sample makes unbiased std warn; we handle it below.
        warnings.simplefilter("ignore")
        std = np.std(y_train, ddof=1)
    # Guard degenerate inputs so the standardization stays finite.
    if not np.isfinite(std) or std == 0.0:
        std = np.float64(1.0)
    else:
        std = std + TARGET_NORM_EPS
    return mean, std


def normalize_targets(y, mean, std):
    """Standardize targets: (y - mean) / std. Works for torch and numpy."""
    return (y - mean) / std


def denormalize_predictions(preds, mean, std):
    """Invert normalize_targets: preds * std + mean. Works for torch and numpy."""
    return preds * std + mean