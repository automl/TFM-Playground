"""Compare previously trained nanoTabPFN models.

Reads saved model folders from trained_models/, lets you pick which to compare,
then runs all the comparison plotting and metrics JSON generation.

Usage:
    python compare_trained_models.py --problem_type classification
    python compare_trained_models.py --problem_type regression --seed 42
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from classification.callback import ClassificationTrackerCallback
from regression.callback import RegressionTrackerCallback
from utils.training import (
    _build_metrics_payload,
    _json_safe,
)
from utils.visualization import (
    plot_comparison_multi,
    plot_all_decision_boundaries,
    plot_all_regression_predictions,
    plot_per_fold_normalized_averaged_metrics,
    plot_per_task_comparison,
    plot_time_budget_metrics,
)
from utils.general import load_config
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.utils import get_default_device


def _discover_trained_models(models_dir: str):
    """Scan models_dir for subdirectories containing metadata.json."""
    models_dir_path = os.path.abspath(models_dir)
    if not os.path.isdir(models_dir_path):
        return []

    found = []
    for entry in sorted(os.listdir(models_dir_path)):
        entry_path = os.path.join(models_dir_path, entry)
        meta_path = os.path.join(entry_path, "metadata.json")
        if os.path.isdir(entry_path) and os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            found.append({
                "dir": entry_path,
                "dir_name": entry,
                "metadata": metadata,
            })
    return found


def _select_models_interactively(available):
    """Let the user pick >= 1 trained models to plot/compare."""
    print("\n" + "=" * 60)
    print("AVAILABLE TRAINED MODELS")
    print("=" * 60)

    for i, item in enumerate(available, 1):
        meta = item["metadata"]
        metric_name = meta.get("metric_name", "metric")
        final_metric = meta.get("final_metric", "?")
        prior_name = meta.get("prior_name", "unknown")
        created = meta.get("created_at", "")
        epochs = meta.get("hyperparams", {}).get("epochs", "?")
        if isinstance(final_metric, float):
            final_metric = f"{final_metric:.4f}"
        print(
            f"  {i:2d}. {item['dir_name']}\n"
            f"      prior: {prior_name}  |  {metric_name}: {final_metric}  |  "
            f"epochs: {epochs}  |  created: {created}"
        )

    print("\n" + "=" * 60)
    print("SELECT MODELS TO COMPARE  (need >= 1)")
    print("=" * 60)
    print("  Enter numbers separated by commas  e.g. 1,3")
    print("  Enter 'all' to compare every model")
    print("  Enter 'quit' to exit")

    while True:
        raw = input("\nYour selection: ").strip().lower()
        if raw == "quit":
            sys.exit(0)
        if raw == "all":
            selected = available
        else:
            try:
                indices = [int(x.strip()) for x in raw.split(",")]
                selected = [
                    available[i - 1] for i in indices if 1 <= i <= len(available)
                ]
            except (ValueError, IndexError):
                print(f"  Invalid input — enter numbers between 1 and {len(available)}.")
                continue

        if len(selected) < 1:
            print("  Please select at least 1 model.")
            continue

        print("\nSelected:")
        for item in selected:
            print(f"  - {item['dir_name']}")
        confirm = input("Proceed? (y/n): ").strip().lower()
        if confirm == "y":
            return selected


def _load_model(model_dir: str, device, is_regression: bool = False):
    """Load a trained NanoTabPFNModel and optional bucket edges from a checkpoint directory."""
    ckpt_path = os.path.join(model_dir, "model.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt["architecture"]
    model = NanoTabPFNModel(
        num_attention_heads=arch["num_attention_heads"],
        embedding_size=arch["embedding_size"],
        mlp_hidden_size=arch["mlp_hidden_size"],
        num_layers=arch["num_layers"],
        num_outputs=arch["num_outputs"],
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Load bucket edges for regression models
    dist = None
    if is_regression:
        bucket_edges_path = os.path.join(model_dir, "bucket_edges.pth")
        if os.path.isfile(bucket_edges_path):
            from pfns.bar_distribution import FullSupportBarDistribution
            bucket_edges = torch.load(bucket_edges_path, map_location=device, weights_only=False)
            dist = FullSupportBarDistribution(bucket_edges).float().to(device)

    return model, dist


def _rebuild_callback(metadata: dict):
    """Rebuild a real tracker callback from saved metadata.

    Creates a ClassificationTrackerCallback or RegressionTrackerCallback
    and populates its attributes from the saved metadata, so that plotting
    functions work exactly as they do after a live training run.
    """
    is_regression = metadata.get("is_regression", False)

    if is_regression:
        callback = RegressionTrackerCallback(tasks=[], model_name="loaded")
        callback.rmse_history = metadata.get("metric_history", [])
        callback.task_rmse_values = metadata.get("per_task_scores", {})
        callback.final_rmse = metadata.get("final_metric", 0.0)
    else:
        callback = ClassificationTrackerCallback(tasks=[], model_name="loaded")
        callback.roc_auc_history = metadata.get("metric_history", [])
        callback.task_roc_auc_values = metadata.get("per_task_scores", {})
        callback.final_roc_auc = metadata.get("final_metric", 0.0)

    callback.epoch_history = metadata.get("epoch_history", [])
    callback.epoch_times = metadata.get("epoch_times", [])
    callback.loss_history = metadata.get("loss_history", [])

    return callback


def main():
    config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))

    parser = argparse.ArgumentParser(
        description="Compare previously trained nanoTabPFN models"
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        choices=["classification", "regression"],
        required=True,
        help="Problem type (determines which trained_models folder to scan).",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=None,
        help="Directory containing trained model folders (default: <problem_type>/results/trained_models/)",
    )
    parser.add_argument(
        "--plot_output",
        type=str,
        default=None,
        help="Path to save the comparison plot (auto-generated if not set)",
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        default=None,
        help="Path to save detailed comparison metrics JSON (auto-generated if not set)",
    )
    parser.add_argument(
        "--seed", type=int, default=config["training"]["seed"], help="Random seed (for decision boundary / regression toy plots)"
    )

    args = parser.parse_args()

    # Defaults
    if args.models_dir is None:
        args.models_dir = os.path.join(
            os.path.dirname(__file__), args.problem_type, "results", "trained_models"
        )

    results_dir = os.path.join(os.path.dirname(__file__), args.problem_type, "results")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.plot_output is None:
        os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
        args.plot_output = os.path.join(results_dir, "plots", f"comparison_{stamp}.png")
    if args.metrics_output is None:
        os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
        args.metrics_output = os.path.join(results_dir, "metrics", f"comparison_{stamp}_metrics.json")

    # Discover and select trained models
    available = _discover_trained_models(args.models_dir)
    if not available:
        print(f"No trained models found in {args.models_dir}")
        print("Run train_single_model.py first to train and save models.")
        sys.exit(1)

    selected = _select_models_interactively(available)

    device = get_default_device()
    print(f"\nUsing device: {device}\n")

    # Check all selected models have the same problem type
    is_regression = selected[0]["metadata"]["is_regression"]
    for item in selected[1:]:
        if item["metadata"]["is_regression"] != is_regression:
            print("ERROR: Cannot compare models with different problem types.")
            sys.exit(1)

    metric_name = "RMSE" if is_regression else "ROC-AUC"

    # Build run_records and callbacks from saved metadata
    run_records = []
    callbacks = []
    prior_names = []

    for idx, item in enumerate(selected, start=1):
        meta = item["metadata"]
        prior_name = meta.get("prior_name", item["dir_name"])

        # Load the actual model for decision boundary / regression prediction plots
        model, dist = _load_model(item["dir"], device, is_regression=is_regression)

        # Rebuild a real callback from saved metadata
        callback = _rebuild_callback(meta)

        prior_names.append(prior_name)
        callbacks.append(callback)

        run_records.append({
            "index": idx,
            "model_name": f"Model {idx}",
            "prior": meta.get("prior_path", ""),
            "prior_name": prior_name,
            "metric": meta.get("final_metric", 0.0),
            "loss_history": meta.get("loss_history", []),
            "metric_history": meta.get("metric_history", []),
            "per_task_scores": meta.get("per_task_scores", {}),
            "train_time": meta.get("train_time", 0.0),
            "inference_time": meta.get("inference_time", 0.0),
            "param_count": meta.get("param_count", 0),
            "model": model,
            "dist": dist,
            "callback": callback,
        })

    # --- Print leaderboard ---
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}\n")

    sorted_runs = sorted(
        run_records, key=lambda r: r["metric"], reverse=not is_regression
    )
    winner = sorted_runs[0]["model_name"] if sorted_runs else None

    print("Leaderboard (sorted by final metric):")
    for rank, r in enumerate(sorted_runs, start=1):
        print(
            f"  {rank:2d}. {r['model_name']}: {r['prior_name']} | "
            f"{metric_name}: {r['metric']:.4f} | "
            f"Train: {r['train_time']:.2f}s | "
            f"Infer: {r['inference_time']*1000:.2f}ms | "
            f"Params: {r['param_count']/1e6:.2f}M"
        )

    print(f"\nWinner: {winner}")
    print(f"\n{'='*80}\n")

    # --- Save metrics JSON ---
    metrics_payload = _build_metrics_payload(
        run_records=run_records,
        metric_name=metric_name,
        is_regression=is_regression,
    )
    metrics_output_dir = os.path.dirname(args.metrics_output)
    if metrics_output_dir:
        os.makedirs(metrics_output_dir, exist_ok=True)
    with open(args.metrics_output, "w", encoding="utf-8") as f:
        json.dump(_json_safe(metrics_payload), f, indent=2)
    print(f"Saved metrics JSON to: {args.metrics_output}")

    # --- Generate all plots ---
    # Change to results dir so _resolve_plot_path (which uses relative "plots/")
    # saves plots under <problem_type>/results/plots/
    os.chdir(results_dir)

    per_fold_output = args.plot_output.replace(".png", "_per_fold_normalized.png")
    plot_per_fold_normalized_averaged_metrics(
        metrics_payload, metric_name=metric_name, output_path=per_fold_output
    )

    plot_comparison_multi(
        callbacks=callbacks,
        prior_names=prior_names,
        save_path=args.plot_output,
        metric_name=metric_name,
    )

    per_task_output = args.plot_output.replace(".png", "_per_task.png")
    plot_per_task_comparison(
        run_records,
        output_path=per_task_output,
        metric_name=metric_name,
    )

    plot_time_budget_metrics(
        run_records,
        metric_name=metric_name,
        output_prefix=os.path.splitext(args.plot_output)[0],
    )

    # Plot decision boundaries or regression predictions
    visual_cfg = config["visualization"]
    if not is_regression:
        decision_boundary_output = args.plot_output.replace(
            ".png", "_decision_boundaries.png"
        )
        plot_all_decision_boundaries(
            run_records,
            datasets=["moons", "circles"],
            n_samples=visual_cfg["classification_toy_samples"],
            noise=visual_cfg["classification_toy_noise"],
            seed=args.seed,
            output_path=decision_boundary_output,
        )
    else:
        regression_output = args.plot_output.replace(
            ".png", "_regression_predictions.png"
        )
        plot_all_regression_predictions(
            run_records,
            datasets=["sinusoidal", "linear", "step"],
            n_samples=visual_cfg["regression_toy_samples"],
            noise=visual_cfg["regression_toy_noise"],
            seed=args.seed,
            output_path=regression_output,
        )

    return {
        "problem_type": "regression" if is_regression else "classification",
        "metric_name": metric_name,
        "winner": winner,
        "runs": run_records,
    }


if __name__ == "__main__":
    results = main()
