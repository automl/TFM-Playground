"""DataLoader and configuration for TabICL-based priors."""

import torch
from tabicl.prior import PriorDataset as TabICLPriorDataset
from tabicl.prior.graph_lib._config import PriorConfig as TabICLPriorDatasetConfig
from torch.utils.data import DataLoader

from tfmplayground.configs.priors import TabICLClassificationPriorConfig, TabICLRegressionPriorConfig
from tfmplayground.priors.base import Prior
from tfmplayground.utils import get_default_device


class TabICLPriorDataLoader(DataLoader):
    """DataLoader sampling synthetic prior data on-the-fly from TabICL's PriorDataset.

    Args:
        num_steps (int): Number of batches to generate per epoch.
        batch_size (int): Number of functions per batch.
        num_datapoints_min (int): Minimum number of datapoints per function.
        num_datapoints_max (int): Maximum number of datapoints per function.
        min_features (int): Minimum number of features in x.
        max_features (int): Maximum number of features in x.
        max_num_classes (int): Maximum number of classes (for classification tasks).
        prior_type (str): Type of prior: 'mlp_scm', 'tree_scm', 'mix_scm' (default), or 'dummy'.
        n_jobs (int): Worker processes the library samples with.
        device (torch.device): Target device for tensors.
    """

    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        num_datapoints_min: int,
        num_datapoints_max: int,
        min_features: int,
        max_features: int,
        max_num_classes: int,
        device: torch.device,
        prior_type: str = "mix_scm",
        n_jobs: int = 1,
    ):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_datapoints_min = num_datapoints_min
        self.num_datapoints_max = num_datapoints_max
        self.min_features = min_features
        self.max_features = max_features
        self.max_num_classes = max_num_classes
        self.prior_type = prior_type
        self.n_jobs = n_jobs
        self.device = device

        self.pd = TabICLPriorDataset(
            batch_size=batch_size,
            batch_size_per_gp=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_num_classes,
            min_seq_len=num_datapoints_min,
            max_seq_len=num_datapoints_max,
            prior_type=prior_type,
            n_jobs=n_jobs,
        )

    def tabicl_to_ours(self, d):
        x, y, active_features, seqlen, train_size = d
        max_active_features = int(active_features.max().item())
        x = x[:, :, :max_active_features]
        train_test_split_index = train_size[0].item()
        return dict(
            x=x.to(self.device),
            y=y.to(self.device),
            target_y=y.to(self.device),  # target_y is identical to y (for downstream compatibility)
            train_test_split_index=train_test_split_index,
        )

    def __iter__(self):
        return iter(self.tabicl_to_ours(next(self.pd)) for _ in range(self.num_steps))

    def __len__(self):
        return self.num_steps


class TabICLPrior(Prior):
    """
    adapts tabicl prior dataset to prior interface
    """

    def __init__(
        self,
        config: TabICLClassificationPriorConfig | TabICLRegressionPriorConfig,
        device: str | torch.device | None = None,
    ) -> None:
        """
        keeps config, and checks datapoint limits
        """
        self.config = config
        self.device = device if device is not None else get_default_device()
        if self.config.num_datapoints_min >= self.config.num_datapoints_max:
            raise ValueError("datapoints minimum must be less than maximum")
        self.built_batch_size = None

    def build_sampler(self, batch_size: int) -> None:
        """
        builds tabicl from config and remembers its batch size
        """
        c = self.config
        self.sampler = TabICLPriorDataset(
            regression=c.problem == "regression",
            batch_size=batch_size,
            batch_size_per_gp=batch_size,
            min_features=c.num_features_min,
            max_features=c.num_features_max,
            max_classes=c.max_num_classes,
            min_seq_len=c.num_datapoints_min,
            max_seq_len=c.num_datapoints_max,
            prior_type=c.prior_type,
            n_jobs=c.n_jobs,
            config=TabICLPriorDatasetConfig(
                filter_unpredictable_datasets=c.filter_unpredictable_datasets,
                filter_unpredictable_graphs=c.filter_unpredictable_graphs,
            ),
        )
        self.built_batch_size = batch_size

    def sample_batch(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        samples one batch of tables with train test split index
        """
        x, y, active_features, _, train_size = next(self.sampler)
        x = x[:, :, : int(active_features.max().item())]
        y = y.float()
        return x.to(self.device), y.to(self.device), train_size[0].item()

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        gives one batch of sampled tables, split at train test index
        """
        if batch_size != self.built_batch_size:
            self.build_sampler(batch_size)
        x, y, sep = self.sample_batch()
        return x[:, :sep], y[:, :sep], x[:, sep:], y[:, sep:]
