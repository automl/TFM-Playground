"""Utility functions for priors."""

from typing import Union

import h5py
import numpy as np
import torch
from tqdm import tqdm
from ticl.priors import GPPrior, MLPPrior

from .config import get_ticl_prior_config


def build_ticl_prior(prior_type: str) -> Union[MLPPrior, GPPrior]:
    """Builds a TICL prior (MLP or GP) based on the prior type string using the defaults in config.py."""

    cfg = get_ticl_prior_config(prior_type)
    if prior_type == "mlp":
        return MLPPrior(cfg)
    elif prior_type == "gp":
        return GPPrior(cfg)
    else:
        raise ValueError(f"Unsupported TICL prior type: {prior_type}")


def dump_prior_to_h5(
    prior, 
    max_classes: int, 
    batch_size: int, 
    save_path: str, 
    problem_type: str, 
    max_seq_len: int, 
    max_features: int
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
        dump_single_eval_pos = f.create_dataset(
            "single_eval_pos", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="i4"
        )

        if problem_type == "classification" and max_classes is not None:
            f.create_dataset("max_num_classes", data=np.array((max_classes,)), chunks=(1,))
        f.create_dataset("original_batch_size", data=np.array((batch_size,)), chunks=(1,))
        f.create_dataset("problem_type", data=problem_type, dtype=h5py.string_dtype())

        for e in tqdm(prior):
            x = e["x"].to("cpu").numpy()
            y = e["y"].to("cpu").numpy()
            single_eval_pos = e["single_eval_pos"]
            if isinstance(single_eval_pos, torch.Tensor):
                single_eval_pos = single_eval_pos.item()

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

            dump_single_eval_pos.resize(dump_single_eval_pos.shape[0] + batch_size, axis=0)
            dump_single_eval_pos[-batch_size:] = single_eval_pos
