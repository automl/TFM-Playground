"""Train nanoTabPFN models on selected priors and save checkpoints + metadata.

Usage:
    python train_single_model.py --problem_type classification --epochs 3 --steps 5
    python train_single_model.py --problem_type regression --epochs 5 --steps 10 --eval_every 2
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from utils.training import (
    _json_safe,
    train_model,
)
from utils.general import discover_h5_files, load_config

from tfmplayground.evaluation import TOY_TASKS_CLASSIFICATION, TOY_TASKS_REGRESSION
from tfmplayground.priors import PriorDumpDataLoader
from tfmplayground.utils import get_default_device, set_randomness_seed


def _select_priors_interactively(problem_type: str):
    """Discover .h5 files for the given problem type and let the user pick which to train."""
    data_dir = os.path.join(os.path.dirname(__file__), problem_type, "results", "data")
    available = discover_h5_files(data_dir)

    if not available:
        print(f"No .h5 files found in {data_dir}")
        print("Run generate_data.py first to generate prior files.")
        sys.exit(1)

    print("\n" + "=" * 50)
    print(f"AVAILABLE PRIORS  ({problem_type.upper()})")
    print("=" * 50)
    prior_list = list(available.items())  # [(name, path), ...]
    for i, (name, path) in enumerate(prior_list, 1):
        print(f"  {i:2d}. {name:<30} {os.path.basename(path)}")

    print("\n" + "=" * 50)
    print("SELECT PRIORS TO TRAIN")
    print("=" * 50)
    print("  Enter numbers separated by commas  e.g. 1,3")
    print("  Enter 'all' to train on every prior")
    print("  Enter 'quit' to exit")

    while True:
        raw = input("\nYour selection: ").strip().lower()
        if raw == "quit":
            sys.exit(0)
        if raw == "all":
            selected = prior_list
        else:
            try:
                indices = [int(x.strip()) for x in raw.split(",")]
                selected = [
                    prior_list[i - 1] for i in indices if 1 <= i <= len(prior_list)
                ]
            except (ValueError, IndexError):
                print(f"  Invalid input — enter numbers between 1 and {len(prior_list)}.")
                continue

        if len(selected) < 1:
            print("  Please select at least 1 prior.")
            continue

        print("\nSelected:")
        for name, path in selected:
            print(f"  - {name}  ({os.path.basename(path)})")
        confirm = input("Proceed? (y/n): ").strip().lower()
        if confirm == "y":
            return selected  # return list of (name, path) tuples


def _resolve_priors_noninteractive(problem_type: str, priors_arg: list):
    """Resolve --priors CLI argument to a list of (name, path) tuples.

    Returns:
        List of (name, path) tuples matching the requested priors.
    """

    data_dir = os.path.join(os.path.dirname(__file__), problem_type, "results", "data")
    available = discover_h5_files(data_dir)

    if not available:
        print(f"No .h5 files found in {data_dir}")
        print("Run generate_data.py first to generate prior files.")
        sys.exit(1)

    if priors_arg == ["all"]:
        selected = list(available.items())
    else:
        unknown = [p for p in priors_arg if p not in available]
        if unknown:
            print(f"ERROR: Unknown prior(s): {unknown}")
            print(f"Available: {list(available.keys())}")
            sys.exit(1)
        selected = [(name, available[name]) for name in priors_arg]

    print(f"\nSelected priors: {[n for n, _ in selected]}")
    return selected


def _save_trained_model(
    output_dir,
    model,
    callback,
    prior_path,
    prior_name,
    problem_type,
    is_regression,
    train_time,
    inference_time,
    param_count,
    metric,
    args,
    criterion=None,
):
    """Save a trained model checkpoint and metadata to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights + architecture
    checkpoint = {
        "model": model.state_dict(),
        "architecture": {
            "num_attention_heads": model.num_attention_heads,
            "embedding_size": model.embedding_size,
            "mlp_hidden_size": model.mlp_hidden_size,
            "num_layers": model.num_layers,
            "num_outputs": model.num_outputs,
        },
    }
    torch.save(checkpoint, os.path.join(output_dir, "model.pth"))

    # Save criterion for regression (raw bucket edges needed to rebuild FullSupportBarDistribution)
    if is_regression and criterion is not None:
        from pfns.bar_distribution import FullSupportBarDistribution
        if isinstance(criterion, FullSupportBarDistribution):
            torch.save(criterion.borders, os.path.join(output_dir, "bucket_edges.pth"))
            print(f"   - bucket_edges.pth")

    # Build metadata with all info the comparison script needs
    metric_history = (
        callback.rmse_history if is_regression else callback.roc_auc_history
    )
    per_task_scores = (
        callback.task_rmse_values if is_regression else callback.task_roc_auc_values
    )

    metadata = {
        "prior_path": prior_path,
        "prior_name": prior_name,
        "problem_type": problem_type,
        "is_regression": is_regression,
        "train_time": train_time,
        "inference_time": inference_time,
        "param_count": param_count,
        "final_metric": metric,
        "metric_name": "RMSE" if is_regression else "ROC-AUC",
        "loss_history": callback.loss_history,
        "metric_history": metric_history,
        "per_task_scores": per_task_scores,
        "epoch_history": callback.epoch_history,
        "epoch_times": callback.epoch_times,
        "hyperparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "lr": args.lr,
            "seed": args.seed,
            "eval_every": args.eval_every,
            "accumulate_gradients": args.accumulate_gradients,
        },
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(metadata), f, indent=2)

    print(f"\n✅ Saved model to: {output_dir}")
    print(f"   - model.pth ({os.path.getsize(os.path.join(output_dir, 'model.pth')) / 1e6:.1f} MB)")
    print(f"   - metadata.json")


def main():
    config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
    train_cfg = config["training"]

    parser = argparse.ArgumentParser(
        description="Train nanoTabPFN models on selected priors and save to disk"
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        choices=["classification", "regression"],
        required=True,
        help="Problem type. Picks priors from <problem_type>/results/data/.",
    )
    parser.add_argument(
        "--epochs", type=int, default=train_cfg["epochs"], help="Number of epochs to train each model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=train_cfg["batch_size"], help="Batch size for training"
    )
    parser.add_argument(
        "--steps", type=int, default=train_cfg["steps"], help="Number of steps per epoch"
    )
    parser.add_argument("--lr", type=float, default=train_cfg["lr"], help="Learning rate")
    parser.add_argument(
        "--accumulate_gradients",
        type=int,
        default=train_cfg["accumulate_gradients"],
        help="Number of gradients to accumulate before updating weights",
    )
    parser.add_argument(
        "--seed", type=int, default=train_cfg["seed"], help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=train_cfg["eval_every"],
        help="Evaluate toy tasks every N epochs",
    )
    parser.add_argument(
        "--toy_tasks_subset",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of OpenML task IDs to evaluate on (space-separated)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base dir for saved models (default: <problem_type>/results/trained_models/)",
    )
    parser.add_argument(
        "--priors",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Prior(s) to train."
            "Use 'all' for every discovered prior, or list names e.g. 'ticl_gp tabpfn_mlp'. "
            "Leave it empty for interactive selection."
        ),
    )

    args = parser.parse_args()

    # Set up output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(__file__), args.problem_type, "results", "trained_models"
        )

    # Select priors: non-interactively when --priors is given, interactive otherwise
    if args.priors is not None:
        selected_priors = _resolve_priors_noninteractive(args.problem_type, args.priors)
    else:
        selected_priors = _select_priors_interactively(args.problem_type)

    # Set random seed
    set_randomness_seed(args.seed)

    # Get device
    device = get_default_device()
    print(f"Using device: {device}\n")

    # Determine which toy tasks to use
    subset = set(args.toy_tasks_subset) if args.toy_tasks_subset else None
    cls_tasks = [t for t in TOY_TASKS_CLASSIFICATION if (subset is None or t in subset)]
    reg_tasks = [t for t in TOY_TASKS_REGRESSION if (subset is None or t in subset)]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for idx, (prior_name, prior_path) in enumerate(selected_priors, start=1):
        # Probe prior to determine problem type
        prior_probe = PriorDumpDataLoader(
            filename=prior_path,
            num_steps=1,
            batch_size=1,
            device=device,
            starting_index=0,
        )
        is_regression = prior_probe.problem_type == "regression"

        # Enforce that probed problem type matches the CLI arg
        if prior_probe.problem_type != args.problem_type:
            print(
                f"ERROR: Prior '{prior_name}' has problem_type='{prior_probe.problem_type}' "
                f"but --problem_type={args.problem_type} was specified."
            )
            sys.exit(1)

        use_tasks = reg_tasks if is_regression else cls_tasks

        # Reset seed for each model for reproducibility
        set_randomness_seed(args.seed)

        model_name = f"Model {idx}"
        trained_model, metric, callback, train_time, inference_time, param_count, criterion = (
            train_model(
                prior_path=prior_path,
                model_name=model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                steps=args.steps,
                lr=args.lr,
                device=device,
                eval_every=args.eval_every,
                tasks=use_tasks,
                accumulate_gradients=args.accumulate_gradients,
                num_attention_heads=config["model"]["num_attention_heads"],
                embedding_size=config["model"]["embedding_size"],
                mlp_hidden_size=config["model"]["mlp_hidden_size"],
                num_layers=config["model"]["num_layers"],
            )
        )

        # Create a folder for this model
        safe_name = prior_name.replace(" ", "_").replace("/", "_")
        model_dir = os.path.join(args.output_dir, f"{safe_name}_{stamp}")

        _save_trained_model(
            output_dir=model_dir,
            model=trained_model,
            callback=callback,
            prior_path=prior_path,
            prior_name=prior_name,
            problem_type=args.problem_type,
            is_regression=is_regression,
            train_time=train_time,
            inference_time=inference_time,
            param_count=param_count,
            metric=metric,
            args=args,
            criterion=criterion,
        )

    print(f"\n{'='*80}")
    print(f"All {len(selected_priors)} model(s) trained and saved to: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
