"""Data loading utilities for priors."""

from typing import Callable, Dict, Iterator, Union

import h5py
import torch
from tabicl.prior.dataset import PriorDataset as TabICLPriorDataset
from ticl.dataloader import PriorDataLoader as TICLPriorDataset
from torch.utils.data import DataLoader


class PriorDataLoader(DataLoader):
    """Generic DataLoader for synthetic data generation using a get_batch function.

    Args:
        get_batch_function (Callable): A function returning batches of data.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Number of functions per batch.
        num_datapoints_max (int): Max sequence length per function.
        num_features (int): Number of input features.
        device (torch.device): Device to move tensors to.
    """

    def __init__(
        self,
        get_batch_function: Callable[..., Dict[str, Union[torch.Tensor, int]]],
        num_steps: int,
        batch_size: int,
        num_datapoints_max: int,
        num_features: int,
        device: torch.device,
    ):
        self.get_batch_function = get_batch_function
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_datapoints_max = num_datapoints_max
        self.num_features = num_features
        self.device = device

    def __iter__(self) -> Iterator[Dict[str, Union[torch.Tensor, int]]]:
        return iter(
            self.get_batch_function(self.batch_size, self.num_datapoints_max, self.num_features)
            for _ in range(self.num_steps)
        )

    def __len__(self) -> int:
        return self.num_steps


class PriorDumpDataLoader(DataLoader):
    """DataLoader that loads synthetic prior data from an HDF5 dump.

    Args:
        filename (str): Path to the HDF5 file.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Batch size.
        device (torch.device): Device to load tensors onto.
    """

    def __init__(self, filename, num_steps, batch_size, device):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        with h5py.File(self.filename, "r") as f:
            if "max_num_classes" in f:
                self.max_num_classes = f["max_num_classes"][0]
            else:
                self.max_num_classes = None
            self.problem_type = f["problem_type"][()].decode("utf-8")
            self.has_num_datapoints = "num_datapoints" in f
            self.stored_max_seq_len = f["X"].shape[1]
        self.device = device
        self.pointer = 0

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size

                num_features = f["num_features"][self.pointer : end].max()
                if self.has_num_datapoints:
                    num_datapoints_batch = f["num_datapoints"][self.pointer:end]
                    max_seq_in_batch = int(num_datapoints_batch.max())
                else:
                    max_seq_in_batch = int(self.stored_max_seq_len)

                x = torch.from_numpy(f["X"][self.pointer:end, :max_seq_in_batch, :num_features])
                y = torch.from_numpy(f["y"][self.pointer:end, :max_seq_in_batch])
                single_eval_pos = f["single_eval_pos"][self.pointer : end]

                self.pointer += self.batch_size
                if self.pointer >= f["X"].shape[0]:
                    print(
                        """Finished iteration over all stored datasets! """
                        """Will start reusing the same data with different splits now."""
                    )
                    self.pointer = 0

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    target_y=y.to(self.device),  # target_y is identical to y (for downstream compatibility)
                    single_eval_pos=single_eval_pos[0].item(),
                )

    def __len__(self):
        return self.num_steps


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


class TICLPriorDataLoader(DataLoader):
    """DataLoader sampling synthetic prior data from TICL's PriorDataLoader.

    Args:
        prior (Any): A TICL prior object supporting get_batch.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Number of functions sampled per batch.
        num_datapoints_max (int): Number of datapoints sampled per function.
        num_features (int): Dimensionality of x vectors.
        device (torch.device): Target device for tensors.
        min_eval_pos (int, optional): Minimum evaluation position in the sequence.
    """

    def __init__(
        self,
        prior,
        num_steps: int,
        batch_size: int,
        num_datapoints_max: int,
        num_features: int,
        device: torch.device,
        min_eval_pos: int = 10,
    ):
        self.num_steps = num_steps
        self.device = device

        self.pd = TICLPriorDataset(
            prior=prior,
            num_steps=num_steps,
            batch_size=batch_size,
            min_eval_pos=min_eval_pos,
            n_samples=num_datapoints_max,
            device=device,
            num_features=num_features,
        )

    def ticl_to_ours(self, d):
        (info, x, y), target_y, single_eval_pos = d
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0)
        target_y = target_y.permute(1, 0)

        return dict(
            x=x.to(self.device),
            y=y.to(self.device),
            target_y=target_y.to(self.device),  # target_y is identical to y (for downstream compatibility)
            single_eval_pos=single_eval_pos,
        )

    def __iter__(self):
        return (self.ticl_to_ours(batch) for batch in self.pd)

    def __len__(self):
        return self.num_steps