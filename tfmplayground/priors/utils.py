"""Utility functions for priors."""

import h5py
import numpy as np
import torch
from ticl.priors import BooleanConjunctionPrior, ClassificationAdapterPrior, GPPrior, MLPPrior, StepFunctionPrior
from tqdm import tqdm

from .config import get_tabpfn_prior_config, get_ticl_prior_config


def build_ticl_prior(
    prior_type: str, base_prior: str = None, max_num_classes: int = None
) -> MLPPrior | GPPrior | ClassificationAdapterPrior | BooleanConjunctionPrior | StepFunctionPrior:
    """Builds a TICL prior based on the prior type string using the defaults in config.py.

    Args:
        prior_type: Type of TICL prior ('mlp', 'gp', 'classification_adapter', etc.)
        base_prior: Base regression prior for composite priors (e.g., 'mlp' or 'gp' for classification_adapter)
        max_num_classes: Maximum number of classes for classification priors
    """

    cfg = get_ticl_prior_config(prior_type)

    if prior_type == "mlp":
        return MLPPrior(cfg)
    elif prior_type == "gp":
        return GPPrior(cfg)
    elif prior_type == "classification_adapter":
        if base_prior is None:
            base_prior = "mlp"  # default to MLP
        # build the base regression prior
        base_prior_obj = build_ticl_prior(base_prior)

        # we equate them rather than treating num_classes as a separate parameter because:
        # - max_num_classes serves as the upper bound for TICL's internal sampling
        # - even with num_classes set to a constant, TICL's class_sampler_f() will internally
        #   vary the actual number of classes (50% chance of 2, 50% chance of uniform(2, num_classes))
        cfg["max_num_classes"] = max_num_classes
        cfg["num_classes"] = max_num_classes
        return ClassificationAdapterPrior(base_prior_obj, **cfg)
    elif prior_type == "boolean_conjunctions":
        return BooleanConjunctionPrior(hyperparameters=cfg)
    elif prior_type == "step_function":
        return StepFunctionPrior(cfg)
    else:
        raise ValueError(f"Unsupported TICL prior type: {prior_type}")


def build_tabpfn_prior(prior_type: str, max_classes: int) -> dict:
    """Builds TabPFN prior configuration with appropriate settings for regression or classification.

    Args:
        prior_type: Type of TabPFN prior ('mlp', 'gp', 'prior_bag')
        max_classes: Maximum number of classes

    Returns:
        dict with 'flexible', 'max_num_classes', and 'prior_config' keys
    """
    is_regression = max_classes == 0

    return {
        "flexible": not is_regression,  # false for regression, true for classification
        "max_num_classes": 2
        if is_regression
        else max_classes,  # library weirdly requires >=2 regardless of regression or classification
        # num_classes parameter in the library code is equated to max_num_classes
        # so its not varied separately here
        "prior_config": {
            **get_tabpfn_prior_config(prior_type),
        },
    }


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
