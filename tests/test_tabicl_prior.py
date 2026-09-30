import pytest
import torch


def test_tabicl_prior_loader_yields_batch():
    """Importing and iterating the TabICL loader must work (regression for the outdated
    tabicl.prior.dataset import path)."""
    pytest.importorskip("tabicl")
    from tfmplayground.external_priors import TabICLPriorDataLoader

    loader = TabICLPriorDataLoader(
        num_steps=1, batch_size=1,
        num_datapoints_min=50, num_datapoints_max=100,
        min_features=5, max_features=10,
        max_num_classes=3, device=torch.device("cpu"),
    )
    batch = next(iter(loader))
    assert "x" in batch and "y" in batch


def test_tabicl_prior_loader_respects_train_size_range():
    """min/max_train_size are forwarded to TabICL: the split stays within [0.3, 0.9] of the rows."""
    pytest.importorskip("tabicl")
    from tfmplayground.external_priors import TabICLPriorDataLoader

    loader = TabICLPriorDataLoader(
        num_steps=5, batch_size=1,
        num_datapoints_min=50, num_datapoints_max=100,
        min_features=5, max_features=10,
        max_num_classes=3, device=torch.device("cpu"),
        log_seq_len=True, min_train_size=0.3, max_train_size=0.9,
    )
    for batch in loader:
        n = batch["x"].shape[1]
        split = batch["train_test_split_index"]
        assert 0.3 * n - 1 <= split <= 0.9 * n + 1