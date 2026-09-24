import numpy as np
import pytest
import torch

from tfmplayground.embedding import leave_one_fold_out_embeddings
from tfmplayground.interface import NanoTabPFNClassifier, NanoTabPFNRegressor
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def _clf():
    torch.manual_seed(0)
    model = NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=3)
    return NanoTabPFNClassifier(model=model, device="cpu")


def _reg():
    torch.manual_seed(0)
    model = NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=8)
    return NanoTabPFNRegressor(model=model, device="cpu")


def test_leave_one_fold_out_embeddings_shape():
    """OOF returns one embedding per TRAINING row, aligned to original order."""
    clf = _clf()
    X = np.random.RandomState(0).randn(20, 3)
    y = np.array([0, 1] * 10)

    oof = leave_one_fold_out_embeddings(clf, X, y, n_folds=5)

    assert oof.shape == (20, 16)


def test_leave_one_fold_out_refits_on_full_data():
    """After extraction the estimator is refit on the full training set, so it stays usable."""
    clf = _clf()
    X = np.random.RandomState(0).randn(20, 3)
    y = np.array([0, 1] * 10)

    leave_one_fold_out_embeddings(clf, X, y, n_folds=5)

    assert len(clf.X_train) == 20


def test_leave_one_fold_out_embeddings_regressor():
    """Works for the regressor too (plain KFold instead of StratifiedKFold)."""
    reg = _reg()
    X = np.random.RandomState(0).randn(20, 3)
    y = np.random.RandomState(1).randn(20)

    oof = leave_one_fold_out_embeddings(reg, X, y, n_folds=5)

    assert oof.shape == (20, 16)


def test_leave_one_fold_out_rejects_too_few_folds():
    """Fewer than 2 folds makes no sense for leave-one-fold-out."""
    clf = _clf()
    X = np.random.RandomState(0).randn(6, 3)
    y = np.array([0, 1, 0, 1, 0, 1])

    with pytest.raises(ValueError):
        leave_one_fold_out_embeddings(clf, X, y, n_folds=1)