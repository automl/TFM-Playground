"""Main module for the priors package."""

import argparse
import random

import numpy as np
import torch

from .dataloader import TabICLPriorDataLoader, TabPFNPriorDataLoader, TICLPriorDataLoader
from .utils import build_tabpfn_prior, build_ticl_prior, dump_prior_to_h5


def main():
    parser = argparse.ArgumentParser(description="Dump prior data (TICL, TabICL, or TabPFN) into HDF5 format.")
    parser.add_argument(
        "--lib",
        type=str,
        required=True,
        choices=["ticl", "tabicl", "tabpfn"],
        help="Which library to use for the prior.",
    )
    parser.add_argument("--save_path", type=str, required=False, help="Path to save the HDF5 file.")
    parser.add_argument("--num_batches", type=int, default=100, help="Number of batches to dump.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for dumping.")
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run prior sampling on."
    )
    parser.add_argument(
        "--prior_type",
        type=str,
        required=True,
        help=(
            "Type of prior to use. For TICL: mlp, gp, classification_adapter, boolean_conjunctions, "
            "step_function. For TabICL: mlp_scm, tree_scm, mix_scm, dummy. For TabPFN: "
            "mlp, gp, prior_bag."
        ),
    )
    parser.add_argument(
        "--base_prior",
        type=str,
        default="mlp",
        choices=["mlp", "gp"],
        help="Base regression prior for composite priors like classification_adapter.",
    )
    parser.add_argument("--min_features", type=int, default=1, help="Minimum number of input features.")
    parser.add_argument("--max_features", type=int, default=100, help="Maximum number of input features.")
    parser.add_argument("--min_seq_len", type=int, default=None, help="Minimum number of data points per function.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum number of data points per function.")
    parser.add_argument("--min_eval_pos", type=int, default=10, help="Minimum evaluation position in the sequence.")
    parser.add_argument(
        "--max_classes",
        type=int,
        default=0,
        help="Maximum number of classes. Set to 0 for regression, >0 for classification.",
    )
    parser.add_argument("--np_seed", type=int, default=None, help="Random seed for NumPy.")
    parser.add_argument("--torch_seed", type=int, default=None, help="Random seed for PyTorch.")

    args = parser.parse_args()

    if args.np_seed is not None:
        np.random.seed(args.np_seed)
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
        random.seed(args.torch_seed)

    device = torch.device(args.device)

    if args.save_path is None:
        args.save_path = (
            f"prior_{args.lib}_{args.prior_type}_"
            f"{args.num_batches}x{args.batch_size}_{args.max_seq_len}x{args.max_features}.h5"
        )

    # infer the problem_type from max_classes
    problem_type = "classification" if args.max_classes > 0 else "regression"

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
    elif args.lib == "tabpfn":
        tabpfn_config = build_tabpfn_prior(args.prior_type, args.max_classes)
        prior = TabPFNPriorDataLoader(
            prior_type=args.prior_type,
            num_steps=args.num_batches,
            batch_size=args.batch_size,
            num_datapoints_max=args.max_seq_len,
            num_features=args.max_features,
            device=device,
            **tabpfn_config,
        )
    else:  # tabicl
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

    dump_prior_to_h5(
        prior, args.max_classes, args.batch_size, args.save_path, problem_type, args.max_seq_len, args.max_features
    )
