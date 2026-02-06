"""Utility functions for priors."""

from typing import Union

import h5py
import numpy as np
import torch
# from ticl.priors import GPPrior, MLPPrior, ClassificationAdapterPrior, BooleanConjunctionPrior, StepFunctionPrior

from .config import get_ticl_prior_config


# def build_ticl_prior(prior_type: str, base_prior_type: str = None, max_num_classes: int = None) -> Union[MLPPrior, GPPrior, ClassificationAdapterPrior, BooleanConjunctionPrior, StepFunctionPrior]:
#     """Builds a TICL prior based on the prior type string using the defaults in config.py."""

#     cfg = get_ticl_prior_config(prior_type, max_num_classes)
    
#     if prior_type == "mlp":
#         return MLPPrior(cfg)
#     elif prior_type == "gp":
#         return GPPrior(cfg)
#     elif prior_type == "classification_adapter":
#         if base_prior_type is None:
#             base_prior_type = "mlp"  # default to MLP
#         # build the base regression prior
#         base_prior = build_ticl_prior(base_prior_type)
#         return ClassificationAdapterPrior(base_prior, **cfg)
#     elif prior_type == "boolean_conjunctions":
#         return BooleanConjunctionPrior(hyperparameters=cfg)
#     elif prior_type == "step_function":
#         return StepFunctionPrior(cfg)
#     else:
#         raise ValueError(f"Unsupported TICL prior type: {prior_type}")


def dump_prior_to_h5(
    prior, 
    max_classes: int, 
    batch_size: int, 
    save_path: str, 
    problem_type: str, 
    max_seq_len: int, 
    max_features: int
):
    """Dumps synthetic prior data into an HDF5 file, sorted by density.
    Optimized to avoid creating a sorted copy of the full dataset in RAM.
    """
    
    print(f"Collecting data for {save_path}...")
    # 1. Buffers (Still necessary to collect data)
    xs, ys, adjs, densities = [], [], [], []
    num_feats_list, num_dps_list, single_eval_pos_list = [], [], []


    # --- Collection Loop (Same as before) ---
    for e in prior:
        x = e["x"].to("cpu").numpy()
        y = e["y"].to("cpu").numpy()
        adj = e['adj'].to('cpu').numpy() 
        density = e['density'].to('cpu').numpy()
        
        sep = e["single_eval_pos"]
        if isinstance(sep, torch.Tensor):
            sep = sep.item()

        current_bs = x.shape[0]
        x_padded = np.pad(
            x, ((0, 0), (0, max_seq_len - x.shape[1]), (0, max_features - x.shape[2])), mode="constant"
        )
        y_padded = np.pad(y, ((0, 0), (0, max_seq_len - y.shape[1])), mode="constant")

        xs.append(x_padded)
        ys.append(y_padded)
        adjs.append(adj)
        densities.append(density)
        
        num_feats_list.append(np.full(current_bs, x.shape[2], dtype="i4"))
        num_dps_list.append(np.full(current_bs, x.shape[1], dtype="i4"))
        single_eval_pos_list.append(np.full(current_bs, sep, dtype="i4"))

    # 2. Concatenate (Creates 1x Memory footprint)
    print("Concatenating arrays...")
    # NOTE: We overwrite the list variable to encourage garbage collection of the list structure
    X_all = np.concatenate(xs, axis=0); del xs
    y_all = np.concatenate(ys, axis=0); del ys
    adj_all = np.concatenate(adjs, axis=0); del adjs
    density_all = np.concatenate(densities, axis=0); del densities
    
    num_features_all = np.concatenate(num_feats_list, axis=0); del num_feats_list
    num_datapoints_all = np.concatenate(num_dps_list, axis=0); del num_dps_list
    single_eval_pos_all = np.concatenate(single_eval_pos_list, axis=0); del single_eval_pos_list

    # 3. Get Sort Indices
    print("Calculating sort indices...")
    sort_indices = np.argsort(density_all)
    
    # 4. Write to HDF5 using indices
    # We create the dataset first, then assign. 
    # This prevents creating `X_all[sort_indices]` (a huge new array) in RAM.
    print("Writing sorted data to disk...")
    max_nodes = max_features + 1
    
    with h5py.File(save_path, "w") as f:
        # Helper to write sorted data
        def write_sorted(name, data_source, shape, chunks, dtype=None):
            dset = f.create_dataset(
                name, shape=shape, maxshape=shape, chunks=chunks, compression="lzf", dtype=dtype
            )
            # WRITING TRICK:
            # Instead of dset[:] = data_source[sort_indices] (which might create a temp copy),
            # we can iterate in chunks if memory is extremely tight, 
            # OR rely on h5py's efficient selection.
            # For most cases, this assignment is optimized by h5py to not fully realize the RHS:
            dset[:] = data_source[sort_indices]

        # Write datasets
        write_sorted("X", X_all, (X_all.shape[0], max_seq_len, max_features), (batch_size, max_seq_len, max_features))
        write_sorted("y", y_all, (y_all.shape[0], max_seq_len), (batch_size, max_seq_len))
        write_sorted("adj", adj_all, (adj_all.shape[0], max_nodes, max_nodes), (batch_size, max_nodes, max_nodes))
        write_sorted("density", density_all, (density_all.shape[0],), (batch_size,))
        write_sorted("num_features", num_features_all, (num_features_all.shape[0],), (batch_size,), dtype="i4")
        write_sorted("num_datapoints", num_datapoints_all, (num_datapoints_all.shape[0],), (batch_size,), dtype="i4")
        write_sorted("single_eval_pos", single_eval_pos_all, (single_eval_pos_all.shape[0],), (batch_size,), dtype="i4")

        # Metadata
        if problem_type == "classification" and max_classes is not None:
            f.create_dataset("max_num_classes", data=np.array((max_classes,)), chunks=(1,))
        f.create_dataset("original_batch_size", data=np.array((batch_size,)), chunks=(1,))
        f.create_dataset("problem_type", data=problem_type, dtype=h5py.string_dtype())

    print("Done.")
