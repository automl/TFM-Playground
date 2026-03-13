"""Utility functions for priors."""

import os
from typing import Union

import h5py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
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
    max_features: int,
    resume: bool = False,
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
        max_nodes = max_features + 1  # +1 for the target node
        dump_adj = f.create_dataset(
            "adj",
            shape=(0, max_nodes, max_nodes),
            maxshape=(None, max_nodes, max_nodes),
            chunks=(batch_size, max_nodes, max_nodes),
        )
        dump_density = f.create_dataset(
            "density", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="f4"
        )

        if problem_type == "classification" and max_classes is not None:
            f.create_dataset("max_num_classes", data=np.array((max_classes,)), chunks=(1,))
        f.create_dataset("original_batch_size", data=np.array((batch_size,)), chunks=(1,))
        f.create_dataset("problem_type", data=problem_type, dtype=h5py.string_dtype())

        for batch_idx, e in tqdm(enumerate(prior), total=len(prior)):
            x = e["x"].to("cpu").numpy()
            y = e["y"].to("cpu").numpy()
            single_eval_pos = e["single_eval_pos"]
            if isinstance(single_eval_pos, torch.Tensor):
                single_eval_pos = single_eval_pos.item()
            adj = e['adj'].to('cpu').numpy() 
            density = e['density'].to('cpu').numpy()

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

            dump_adj.resize(dump_adj.shape[0] + batch_size, axis=0) 
            dump_adj[-batch_size:] = adj

            dump_density.resize(dump_density.shape[0] + batch_size, axis=0)
            dump_density[-batch_size:] = density

            # Periodic flush for crash safety
            if (batch_idx + 1) % 50 == 0:
                f.flush()

def sort_h5_by_density(
    input_path: str,
    output_path: str,
    chunk_size: int = 10_000,
):
    """
    Sorts an HDF5 file by density without loading the entire dataset into memory.
    
    Args:
        input_path: Path to the unsorted HDF5 file (must contain 'density' dataset)
        output_path: Path for the sorted output file
        chunk_size: Number of samples to process at a time (controls memory usage). Adjust based on your system's memory constraints.
    """
    
    if input_path == output_path:
        raise ValueError("Input and output paths must be different")
    
    with h5py.File(input_path, "r") as f_in:
        # 1. Load only density to compute sort indices (small memory footprint)
        print("Loading density values and computing sort order...")
        density = f_in["density"][:]
        total_samples = len(density)
        sort_indices = np.argsort(density)
        del density  # Free memory
        
        print(f"Total samples: {total_samples}")
        
        # 2. Create output file with same structure
        with h5py.File(output_path, "w") as f_out:
            # Copy metadata datasets directly
            metadata_keys = ["max_num_classes", "original_batch_size", "problem_type"]
            for key in metadata_keys:
                if key in f_in:
                    if key == "problem_type":
                        f_out.create_dataset(key, data=f_in[key][()], dtype=h5py.string_dtype())
                    else:
                        f_out.create_dataset(key, data=f_in[key][:])
            
            # Get data datasets (exclude metadata)
            data_keys = [k for k in f_in.keys() if k not in metadata_keys]
            
            # Create output datasets with same shape/dtype
            out_datasets = {}
            for key in data_keys:
                src = f_in[key]
                out_datasets[key] = f_out.create_dataset(
                    key,
                    shape=src.shape,
                    dtype=src.dtype,
                    chunks=src.chunks,
                    compression=src.compression,
                )
            
            # 3. Process in chunks to limit memory usage
            print("Writing sorted data in chunks...")
            
            # We'll iterate through OUTPUT positions in chunks
            # For each output chunk, we need to gather from scattered input positions
            for out_start in tqdm(range(0, total_samples, chunk_size)):
                out_end = min(out_start + chunk_size, total_samples)
                
                # Which input indices map to this output chunk?
                chunk_input_indices = sort_indices[out_start:out_end]
                
                # For efficient HDF5 reading, sort the input indices
                # (HDF5 reads are faster with sorted/contiguous indices)
                read_order = np.argsort(chunk_input_indices)
                sorted_input_indices = chunk_input_indices[read_order]
                
                # Compute inverse to restore original order after reading
                inverse_order = np.argsort(read_order)
                
                # Read and write each dataset
                for key in data_keys:
                    # Read from scattered input positions (sorted for efficiency)
                    # Convert to list for h5py fancy indexing
                    data_chunk = f_in[key][sorted_input_indices.tolist()]
                    
                    # Reorder to match output order
                    data_chunk = data_chunk[inverse_order]
                    
                    # Write to contiguous output positions
                    out_datasets[key][out_start:out_end] = data_chunk
                
                # Periodic flush
                if (out_start // chunk_size) % 50 == 0:
                    f_out.flush()
            
            f_out.flush()
    
    print(f"Done. Sorted file saved to: {output_path}")


def slice_h5_by_density(
    input_path: str,
    output_path: str,
    min_density: float ,
    max_density: float,
    n_samples: int,
    random_seed: int = 42,
    chunk_size: int = 10_000,
):
    """Creates a filtered, shuffled subset of an HDF5 prior dump file.

    Replicates the filtering logic of the old PriorDumpDataLoader at file-creation
    time so the simple sequential PriorDumpDataLoader produces identical data.

    The output file contains only the selected samples written in shuffled order,
    ready for sequential reading.

    Args:
        input_path: Path to the source HDF5 file. Must be sorted by density
                    (run sort_h5_by_density first).
        output_path: Path for the filtered output file.
        min_density: Only include samples with density >= this value. Must be in [0, 1].
        max_density: Only include samples with density <= this value. Must be in [0, 1].
        n_samples: Number of samples to randomly select from the density-filtered range.
        random_seed: Random seed for reproducible sampling and shuffling. Defaults to 42.
        chunk_size: Number of samples to copy at a time (controls memory usage).
    """
    if input_path == output_path:
        raise ValueError("input_path and output_path must be different")
    
    min_density = np.float32(min_density)
    max_density = np.float32(max_density)

    if not (0 <= max_density <= 1):
        raise ValueError(f"max_density must be between 0 and 1, got {max_density}")
    if not (0 <= min_density <= 1):
        raise ValueError(f"min_density must be between 0 and 1, got {min_density}")
    if min_density > max_density:
        raise ValueError(
            f"min_density ({min_density}) cannot be greater than max_density ({max_density})"
        )

    # --- Compute the shuffled indices (mirrors old PriorDumpDataLoader logic exactly) ---
    with h5py.File(input_path, "r") as f_in:
        all_densities = f_in["density"][:]

        if not np.all(all_densities[:-1] <= all_densities[1:]):
            raise ValueError(
                f"'{input_path}' is not sorted by density. "
                "Run sort_h5_by_density first."
            )
        
        if min_density == 0.:
            start_idx = 0
        else:
            start_idx = int(np.searchsorted(all_densities, min_density, side="left"))

        if max_density == 1.:
            end_idx = len(all_densities)
        else:
            end_idx = int(np.searchsorted(all_densities, max_density, side="right"))

        if start_idx >= end_idx:
            raise ValueError(
                f"No samples found with density in [{min_density}, {max_density}]. "
                f"Available range: [{all_densities.min():.4f}, {all_densities.max():.4f}]"
            )

        candidate_indices = np.arange(start_idx, end_idx)
        num_candidates = len(candidate_indices)

        if n_samples > num_candidates:
            raise ValueError(
                f"Insufficient samples after density filtering: "
                f"found {num_candidates}, but n_samples={n_samples}. "
                f"Density range: [{min_density}, {max_density}]."
            )
        rng = np.random.RandomState(random_seed)
        selected_indices = rng.choice(candidate_indices, size=n_samples, replace=False)
        rng.shuffle(selected_indices)

        densities_selected = all_densities[selected_indices]
        print(
            f"Density filter active: selected {len(selected_indices)} samples "
            f"with density range [{densities_selected.min():.4f}, {densities_selected.max():.4f}]"
        )

        num_output = len(selected_indices)
        print(f"Writing {num_output} samples to '{output_path}'...")

        metadata_keys = ["max_num_classes", "original_batch_size", "problem_type"]
        data_keys = [k for k in f_in.keys() if k not in metadata_keys]

        with h5py.File(output_path, "w") as f_out:
            # Copy scalar/metadata datasets unchanged
            for key in metadata_keys:
                if key in f_in:
                    if key == "problem_type":
                        f_out.create_dataset(key, data=f_in[key][()], dtype=h5py.string_dtype())
                    else:
                        f_out.create_dataset(key, data=f_in[key][:])

            # Create output datasets with the reduced first dimension
            out_datasets = {}
            for key in data_keys:
                src = f_in[key]
                new_shape = (num_output,) + src.shape[1:]
                out_datasets[key] = f_out.create_dataset(
                    key,
                    shape=new_shape,
                    dtype=src.dtype,
                    chunks=(min(src.chunks[0], num_output),) + src.chunks[1:] if src.chunks else None,
                    compression=src.compression,
                )

            # Write selected samples in chunks
            for out_start in tqdm(range(0, num_output, chunk_size)):
                out_end = min(out_start + chunk_size, num_output)
                chunk_input_indices = selected_indices[out_start:out_end]

                # Sort input indices for efficient HDF5 reads, then restore order
                read_order = np.argsort(chunk_input_indices)
                sorted_input_indices = chunk_input_indices[read_order]
                inverse_order = np.argsort(read_order)

                for key in data_keys:
                    data_chunk = f_in[key][sorted_input_indices.tolist()]
                    out_datasets[key][out_start:out_end] = data_chunk[inverse_order]

                if (out_start // chunk_size) % 50 == 0:
                    f_out.flush()

            f_out.flush()

    print(f"Done. Filtered file saved to: {output_path}")
