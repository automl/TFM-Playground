"""DataLoader and configuration for TabICL-based priors."""

import torch
from tabicl.prior import PriorDataset as TabICLPriorDataset
from torch.utils.data import DataLoader


class TabICLPriorDataLoader(DataLoader):
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
        log_seq_len: bool = False,        # NUEVO
        min_train_size: float = 0.1,      # NUEVO
        max_train_size: float = 0.9,      # NUEVO
    ):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_datapoints_min = num_datapoints_min
        self.num_datapoints_max = num_datapoints_max
        self.min_features = min_features
        self.max_features = max_features
        self.max_num_classes = max_num_classes
        self.prior_type = prior_type
        self.device = device

        self.pd = TabICLPriorDataset(
            batch_size=batch_size,
            batch_size_per_gp=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_num_classes,
            min_seq_len=num_datapoints_min,
            max_seq_len=num_datapoints_max,
            log_seq_len=log_seq_len,          # NUEVO
            min_train_size=min_train_size,    # NUEVO
            max_train_size=max_train_size,    # NUEVO
            prior_type=prior_type,
            n_jobs=1,
        )

    def tabicl_to_ours(self, d):
        x, y, active_features, seqlen, train_size = d
        active_features = active_features[0].item()
        x = x[:, :, :active_features]
        train_test_split_index = train_size[0].item()
        # graph_scm returns the class labels as int64 (mlp_scm/tree_scm as float); the model averages
        # the train labels (pad_targets), so hand float targets to train() for every prior type.
        y = y.float()
        return dict(
            x=x.to(self.device),
            y=y.to(self.device),
            target_y=y.to(self.device),
            train_test_split_index=train_test_split_index,
        )

    def __iter__(self):
        return iter(self.tabicl_to_ours(next(self.pd)) for _ in range(self.num_steps))

    def __len__(self):
        return self.num_steps