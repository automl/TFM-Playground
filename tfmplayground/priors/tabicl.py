import torch
from tabicl.prior.dataset import PriorDataset as TabICLPriorDataset
from torch.utils.data import DataLoader


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
    ):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_datapoints_min = num_datapoints_min
        self.num_datapoints_max = num_datapoints_max
        self.min_features = min_features
        self.max_features = max_features
        self.max_num_classes = max_num_classes
        self.device = device

        self.pd = TabICLPriorDataset(
            batch_size=batch_size,
            batch_size_per_gp=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_num_classes,
            min_seq_len=num_datapoints_min,
            max_seq_len=num_datapoints_max,
        )

    def tabicl_to_ours(self, d):
        x, y, active_features, seqlen, train_size = d
        active_features = active_features[
            0
        ].item()  # should be all the same since we use batch_size_per_gp=batch_size (not true in practice!)
        x = x[:, :, :active_features]
        single_eval_pos = train_size[0].item()  # should be all the same since we use batch_size_per_gp=batch_size
        return dict(
            x=x.to(self.device),
            y=y.to(self.device),
            target_y=y.to(self.device),  # target_y is identical to y (for downstream compatibility)
            single_eval_pos=single_eval_pos,
        )

    def __iter__(self):
        return iter(self.tabicl_to_ours(next(self.pd)) for _ in range(self.num_steps))

    def __len__(self):
        return self.num_steps