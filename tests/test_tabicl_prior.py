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


def test_tabicl_to_ours_casts_integer_labels_to_float():
    """graph_scm returns class labels as int64; the loader must hand float targets to train()
    like the other prior types, because the model averages them (pad_targets)."""
    pytest.importorskip("tabicl")
    from tfmplayground.external_priors import TabICLPriorDataLoader

    loader = TabICLPriorDataLoader.__new__(TabICLPriorDataLoader)  # no prior generation needed
    loader.device = torch.device("cpu")
    x = torch.randn(2, 10, 5)
    y = torch.randint(0, 3, (2, 10))  # int64, as graph_scm returns
    d = (x, y, torch.tensor([5, 5]), torch.tensor([10, 10]), torch.tensor([6, 6]))

    out = loader.tabicl_to_ours(d)

    assert out["y"].dtype == torch.float32 and out["target_y"].dtype == torch.float32
    assert torch.equal(out["y"], y.float())


def _loader_with_batches(batches):
    """A TabICLPriorDataLoader whose TabICL generator is replaced by the given raw batches."""
    from tfmplayground.external_priors import TabICLPriorDataLoader

    loader = TabICLPriorDataLoader.__new__(TabICLPriorDataLoader)
    loader.device = torch.device("cpu")
    loader.num_steps = 1
    loader.pd = iter(batches)
    return loader


def test_loader_skips_batches_without_usable_datasets_and_drops_bad_ones():
    """A batch where every dataset has 0 features is skipped; in the next one, the dataset with
    0 features and the failed one (labels -100) are dropped and only the good one is returned."""
    pytest.importorskip("tabicl")
    x = torch.randn(3, 10, 6)
    y = torch.randint(0, 3, (3, 10)).float()
    y_failed = y.clone()
    y_failed[2] = -100.0
    all_empty = (x[:2], y[:2], torch.tensor([0, 0]), torch.tensor([10, 10]), torch.tensor([6, 6]))
    mixed = (x, y_failed, torch.tensor([0, 4, 5]), torch.tensor([10, 10, 10]), torch.tensor([6, 6, 6]))

    batch = next(iter(_loader_with_batches([all_empty, mixed])))

    assert batch["x"].shape == (1, 10, 4)  # only dataset 1 survives, with its 4 features
    assert torch.equal(batch["y"][0], y[1])


def test_loader_keeps_features_of_the_widest_dataset():
    """With different feature counts in a batch, no dataset is truncated: x keeps the widest."""
    pytest.importorskip("tabicl")
    x = torch.randn(2, 10, 8)
    y = torch.randint(0, 3, (2, 10)).float()
    batch = (x, y, torch.tensor([3, 7]), torch.tensor([10, 10]), torch.tensor([6, 6]))

    out = next(iter(_loader_with_batches([batch])))

    assert out["x"].shape == (2, 10, 7)
