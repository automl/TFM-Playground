"""Utility functions for priors."""

from typing import Union

import h5py
import numpy as np
import torch
from ticl.priors import GPPrior, MLPPrior, ClassificationAdapterPrior, BooleanConjunctionPrior, StepFunctionPrior

from .config import get_ticl_prior_config, get_tabpfn_prior_config, get_priors_for_lib


def validate_prior_config(lib: str, prior_type: str = None, max_classes: int = 0):
    """Validate that prior_type is compatible with the selected library.
    
    Args:
        lib: Library name (e.g., 'ticl', 'tabicl', 'tabpfn')
        prior_type: Type of prior to use (None to use dataloader defaults)
        max_classes: Maximum number of classes (required > 0 for classification only libraries)
        
    Raises:
        ValueError: If prior_type is not compatible with the library or max_classes is invalid.
    """
    # if prior_type is None, skip validation (dataloaders will use their defaults)
    if prior_type is None:
        return
    
    available_priors = get_priors_for_lib(lib)
    
    if prior_type not in available_priors:
        raise ValueError(
            f"Unknown {lib.upper()} prior: '{prior_type}'. "
            f"Available {lib.upper()} priors: {', '.join(available_priors)}"
        )
    
    # TabICL is classification-only, requires max_classes > 0
    if lib == "tabicl" and max_classes <= 0:
        raise ValueError(
            f"TabICL is classification-only and requires --max_classes > 0. "
            f"Current value: {max_classes}. Set --max_classes to a positive integer (e.g., 10)."
        )
    #note: there is no base_prior validation - dataloaders handle their own defaults


def build_ticl_prior(prior_type: str, base_prior: str = None, max_num_classes: int = None) -> Union[MLPPrior, GPPrior, ClassificationAdapterPrior, BooleanConjunctionPrior, StepFunctionPrior]:
    """Builds a TICL prior based on the prior type string using the defaults in config.py.
    
    Args:
        prior_type: Type of TICL prior ('mlp', 'gp', 'classification_adapter', etc.)
        base_prior: Base regression prior for composite priors (e.g., 'mlp' or 'gp' for classification_adapter)
        max_num_classes: Maximum number of classes for classification priors
    """

    cfg = get_ticl_prior_config(prior_type, max_num_classes)
    
    if prior_type == "mlp":
        return MLPPrior(cfg)
    elif prior_type == "gp":
        return GPPrior(cfg)
    elif prior_type == "classification_adapter":
        if base_prior is None:
            base_prior = "mlp"  # default to MLP
        # build the base regression prior
        base_prior_obj = build_ticl_prior(base_prior)
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
        max_classes: Maximum number of classes (0 for regression, >0 for classification)
        
    Returns:
        dict with 'flexible', 'max_num_classes', and 'prior_config' keys
    """
    is_regression = max_classes == 0
    
    return {
        'flexible': not is_regression,  # false for regression, true for classification
        'max_num_classes': 2 if is_regression else max_classes,  # library requires >=2 regardless of regression or classification
        'prior_config': get_tabpfn_prior_config(prior_type, max_classes),
    }


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

        for e in prior:
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
