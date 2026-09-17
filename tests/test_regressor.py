"""Interface tests for NanoTabPFNRegressor."""

import numpy as np
import torch

from tfmplayground.interface import NanoTabPFNRegressor

X = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])


def _make_regressor():
    # Identity model + explicit cpu so __init__ neither downloads a checkpoint
    # nor needs a GPU (dist stays None; fit does not use it).
    return NanoTabPFNRegressor(model=torch.nn.Identity(), device="cpu")


def test_fit_returns_self():
    """scikit-learn convention: fit returns the estimator, so calls can chain."""
    reg = _make_regressor()
    assert reg.fit(X, np.array([1.5, 2.5, 3.5])) is reg