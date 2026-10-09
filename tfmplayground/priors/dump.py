import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tfmplayground.configs.priors import PriorDumpConfig
from tfmplayground.priors.base import Prior
from tfmplayground.utils import get_default_device


class DumpPrior(Prior):
    """
    adapts h5 dumps of prior batches to prior interface
    """

    def __init__(
        self,
        config: PriorDumpConfig,
        device: str | torch.device | None = None,
    ) -> None:
        """
        reads sizes and keys from dump and checks its problem against config
        """
        self.config = config
        with h5py.File(config.filename, "r") as f:
            self.num_tables = f["X"].shape[0]
            self.num_datapoints_max = f["X"].shape[1]
            self.has_num_datapoints = "num_datapoints" in f
            self.sep_key = "train_test_split_index" if "train_test_split_index" in f else "single_eval_pos"
            self.block_size = int(f["original_batch_size"][0]) if "original_batch_size" in f else None
            self.problem = f["problem_type"][()].decode("utf-8") if "problem_type" in f else None
        config_problem = getattr(config, "problem", None)
        if self.problem is not None and config_problem is not None and self.problem != config_problem:
            raise ValueError(f"dump holds {self.problem!r} but config says {config_problem!r}")
        self.device = device if device is not None else get_default_device()
        self.pointer = config.starting_index

    def check_batch_size(self, batch_size: int) -> None:
        """
        checks that every batch stays inside one dump block
        """
        if batch_size > self.num_tables:
            raise ValueError(f"batch size {batch_size} is larger than {self.num_tables} tables in dump")
        if self.block_size is not None and self.block_size % batch_size:
            raise ValueError(f"batch size {batch_size} does not divide dump block size {self.block_size}")
        if self.block_size is not None and self.pointer % batch_size:
            raise ValueError(f"starting index {self.pointer} does not fit batch size {batch_size}")

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        gives next batch of tables from dump, split into train and test parts

        starts dump again at its end, so batches never run out
        """
        self.check_batch_size(batch_size)
        if self.pointer + batch_size > self.num_tables:
            self.pointer = 0
        with h5py.File(self.config.filename, "r") as f:
            end = self.pointer + batch_size
            num_features = f["num_features"][self.pointer : end].max()
            if self.has_num_datapoints:
                max_seq_in_batch = int(f["num_datapoints"][self.pointer : end].max())
            else:
                max_seq_in_batch = int(self.num_datapoints_max)
            x = torch.from_numpy(f["X"][self.pointer : end, :max_seq_in_batch, :num_features])
            y = torch.from_numpy(f["y"][self.pointer : end, :max_seq_in_batch])
            splits = f[self.sep_key][self.pointer : end]
            if splits.min() != splits.max():
                raise ValueError(f"batch has different train test splits in range {splits.min()} to {splits.max()}")
            sep = int(splits[0])
            self.pointer += batch_size
        loaded_batch_size = x.shape[0]
        if loaded_batch_size != batch_size:
            raise ValueError(f"batch size is {batch_size} but dump gives {loaded_batch_size}")
        x = x.to(self.device)
        y = y.to(self.device)
        return x[:, :sep], y[:, :sep], x[:, sep:], y[:, sep:]


class PriorDumpDataLoader(DataLoader):
    """DataLoader that loads synthetic prior data from an HDF5 dump.

    Args:
        filename (str): Path to the HDF5 file.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Batch size.
        device (torch.device): Device to load tensors onto.
    """

    def __init__(self, filename, num_steps, batch_size, device, starting_index=0):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        with h5py.File(self.filename, "r") as f:
            self.num_tables = f["X"].shape[0]
            self.num_datapoints_max = f["X"].shape[1]
            self.num_features_max = f["X"].shape[2]
            if "max_num_classes" in f:
                self.max_num_classes = f["max_num_classes"][0]
            else:
                self.max_num_classes = None
            self.problem_type = f["problem_type"][()].decode("utf-8")
            self.has_num_datapoints = "num_datapoints" in f
            self.sep_key = "train_test_split_index" if "train_test_split_index" in f else "single_eval_pos"
        self.device = device
        self.pointer = starting_index

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size

                num_features = f["num_features"][self.pointer : end].max()
                if self.has_num_datapoints:
                    num_datapoints_batch = f["num_datapoints"][self.pointer : end]
                    max_seq_in_batch = int(num_datapoints_batch.max())
                else:
                    max_seq_in_batch = int(self.num_datapoints_max)

                x = torch.from_numpy(f["X"][self.pointer : end, :max_seq_in_batch, :num_features])
                y = torch.from_numpy(f["y"][self.pointer : end, :max_seq_in_batch])
                train_test_split_index = f[self.sep_key][self.pointer : end]

                self.pointer += self.batch_size
                if self.pointer >= self.num_tables:
                    print(
                        """Finished iteration over all stored datasets! """
                        """Will start reusing the same data with different splits now."""
                    )
                    self.pointer = 0

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    target_y=y.to(self.device),  # target_y is identical to y (for downstream compatibility)
                    train_test_split_index=train_test_split_index[0].item(),
                )

    def __len__(self):
        return self.num_steps


def dump_prior_to_h5(
    prior, max_classes: int, batch_size: int, save_path: str, problem_type: str, max_seq_len: int, max_features: int
):
    """Dumps synthetic prior data into an HDF5 file for later training."""

    with h5py.File(save_path, "w") as f:
        dump_X = f.create_dataset(
            "X",
            shape=(0, max_seq_len, max_features),
            maxshape=(None, max_seq_len, max_features),
            chunks=(batch_size, max_seq_len, max_features),
            compression="lzf",
        )
        dump_num_features = f.create_dataset(
            "num_features", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="i4"
        )
        dump_num_datapoints = f.create_dataset(
            "num_datapoints", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="i4"
        )
        dump_y = f.create_dataset(
            "y", shape=(0, max_seq_len), maxshape=(None, max_seq_len), chunks=(batch_size, max_seq_len)
        )
        dump_train_test_split_index = f.create_dataset(
            "train_test_split_index", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="i4"
        )

        if problem_type == "classification":
            f.create_dataset("max_num_classes", data=np.array((max_classes,)), chunks=(1,))
        f.create_dataset("original_batch_size", data=np.array((batch_size,)), chunks=(1,))
        f.create_dataset("problem_type", data=problem_type, dtype=h5py.string_dtype())

        for e in tqdm(prior):
            x = e["x"].to("cpu").numpy()
            y = e["y"].to("cpu").numpy()
            train_test_split_index = e["train_test_split_index"]
            if isinstance(train_test_split_index, torch.Tensor):
                train_test_split_index = train_test_split_index.item()

            # pad x and y to the maximum sequence length and number of features needed for tabicl
            x_padded = np.pad(
                x, ((0, 0), (0, max_seq_len - x.shape[1]), (0, max_features - x.shape[2])), mode="constant"
            )
            y_padded = np.pad(y, ((0, 0), (0, max_seq_len - y.shape[1])), mode="constant")

            dump_X.resize(dump_X.shape[0] + batch_size, axis=0)
            dump_X[-batch_size:] = x_padded

            dump_y.resize(dump_y.shape[0] + batch_size, axis=0)
            dump_y[-batch_size:] = y_padded

            dump_num_features.resize(dump_num_features.shape[0] + batch_size, axis=0)
            dump_num_features[-batch_size:] = x.shape[2]

            dump_num_datapoints.resize(dump_num_datapoints.shape[0] + batch_size, axis=0)
            dump_num_datapoints[-batch_size:] = x.shape[1]

            dump_train_test_split_index.resize(dump_train_test_split_index.shape[0] + batch_size, axis=0)
            dump_train_test_split_index[-batch_size:] = train_test_split_index
