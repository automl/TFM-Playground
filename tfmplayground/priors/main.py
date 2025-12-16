"""Main module for the priors package."""

import argparse
import random

import numpy as np
import torch

from .dataloader import TabICLPriorDataLoader, TICLPriorDataLoader
from .utils import build_ticl_prior, dump_prior_to_h5, validate_prior_config
from .config import get_available_libraries, is_classification_prior, get_default_prior


def main():
    parser = argparse.ArgumentParser(description="Dump TICL or TabICL prior into HDF5 format.")
    parser.add_argument("--lib", type=str, required=True, choices=get_available_libraries(), help="Which library to use for the prior.")
    parser.add_argument("--save_path", type=str, required=False, help="Path to save the HDF5 file.")
    parser.add_argument("--num_batches", type=int, default=100, help="Number of batches to dump.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for dumping.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run prior sampling on.")
    parser.add_argument("--prior_type", type=str, required=False, help="Type of prior to use. For TICL: mlp, gp, classification_adapter, boolean_conjunctions, step_function. For TabICL: mlp_scm, tree_scm, mix_scm, dummy.")
    parser.add_argument("--base_prior", type=str, default="mlp", choices=["mlp", "gp"], help="Base regression prior for composite priors like classification_adapter.")
    parser.add_argument("--min_features", type=int, default=1, help="Minimum number of input features.")
    parser.add_argument("--max_features", type=int, default=100, help="Maximum number of input features.")
    parser.add_argument("--min_seq_len", type=int, default=None, help="Minimum number of data points per function.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum number of data points per function.")
    parser.add_argument("--min_eval_pos", type=int, default=10, help="Minimum evaluation position in the sequence.")
    parser.add_argument("--max_classes", type=int, default=10, help="Maximum number of classes (classification only).")
    parser.add_argument("--np_seed", type=int, default=None, help="Random seed for NumPy.")
    parser.add_argument("--torch_seed", type=int, default=None, help="Random seed for PyTorch.")

    args = parser.parse_args()

    # validate the prior configuration
    validate_prior_config(args.lib, args.prior_type, args.base_prior)

    # Set prior_type to default if not provided
    if args.prior_type is None:
        args.prior_type = get_default_prior(args.lib)

    if args.np_seed is not None:
        np.random.seed(args.np_seed)
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
        random.seed(args.torch_seed)

    device = torch.device(args.device)

    if args.save_path is None:
        args.save_path = f"prior_{args.lib}_{args.prior_type}_{args.num_batches}x{args.batch_size}_{args.max_seq_len}x{args.max_features}.h5"

    # determine if this is a classification prior
    problem_type = "classification" if is_classification_prior(args.lib, args.prior_type) else "regression"

    if args.lib == "ticl":
        prior = TICLPriorDataLoader(
            prior=build_ticl_prior(args.prior_type, args.base_prior, args.max_classes),
            num_steps=args.num_batches,
            batch_size=args.batch_size,
            num_datapoints_max=args.max_seq_len,
            num_features=args.max_features,
            device=device,
            min_eval_pos=args.min_eval_pos,
        )
    else:
        if args.min_seq_len == args.max_seq_len:
            args.min_seq_len = None  # TabICL prior requires min_seq_len < max_seq_len
        
        prior = TabICLPriorDataLoader(
            num_steps=args.num_batches,
            batch_size=args.batch_size,
            num_datapoints_min=args.min_seq_len,
            num_datapoints_max=args.max_seq_len,
            min_features=args.min_features,
            max_features=args.max_features,
            max_num_classes=args.max_classes,
            prior_type=args.prior_type,
            device=device,
        )

    dump_prior_to_h5(prior, args.max_classes, args.batch_size, args.save_path, problem_type, args.max_seq_len, args.max_features)
