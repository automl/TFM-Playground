"""Compare previously trained nanoTabPFN models.

Reads saved model folders from trained_models/, lets you pick which to compare,
then runs all the comparison plotting and metrics JSON generation.

Usage:
    python compare_trained_models.py --problem_type classification
    python compare_trained_models.py --problem_type regression --seed 42
    python compare_trained_models.py --problem_type classification --models all
"""

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, r2_score

from tfmplayground.priors.experiments.classification.callback import (
    ClassificationTrackerCallback,
)
from tfmplayground.priors.experiments.regression.callback import (
    RegressionTrackerCallback,
)
from tfmplayground.priors.experiments.experiment_evaluation import (
    get_openml_predictions,
    TABARENA_TASKS,
)
from tfmplayground.priors.experiments.base_analyzer import (
    compute_summary_similarity_matrix,
)
from tfmplayground.priors.experiments.classification.analyzer import (
    ClassificationDataAnalyzer,
)
from tfmplayground.priors.experiments.regression.analyzer import RegressionDataAnalyzer
from tfmplayground.priors.experiments.utils.training import (
    _build_metrics_payload,
    _json_safe,
)
from tfmplayground.priors.experiments.utils.visualization import (
    plot_comparison_multi,
    plot_all_decision_boundaries,
    plot_all_regression_predictions,
    plot_per_fold_normalized_averaged_metrics,
    plot_per_task_comparison,
    plot_time_budget_metrics,
    plot_tabarena_performance_heatmap,
    plot_tabarena_normalized_heatmap,
    plot_prior_correlation_heatmap,
    compute_performance_similarity_matrix,
    plot_data_similarity_heatmap,
    plot_data_vs_performance_similarity,
    plot_small_data_and_performance_similarity_heatmaps,
)
from tfmplayground.priors.experiments.utils.general import load_config
from tfmplayground.priors.experiments.utils.general import (
    discover_h5_files,
    discover_trained_models,
)
from tfmplayground.interface import NanoTabPFNClassifier, NanoTabPFNRegressor
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.utils import get_default_device


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
                print(
                    f"  Invalid input — enter numbers between 1 and {len(available)}."
                )
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


def _select_models_noninteractive(available, models_arg):
    """Resolve --models CLI argument to a list of model items.

    Args:
        available: List of discovered model dicts from discover_trained_models.
        models_arg: List of model directory names, or ['all'].

    Returns:
        List of selected model dicts.
    """
    if models_arg == ["all"]:
        selected = available
    else:
        name_to_item = {item["dir_name"]: item for item in available}
        unknown = [m for m in models_arg if m not in name_to_item]
        if unknown:
            print(f"ERROR: Unknown model(s): {unknown}")
            print(f"Available: {list(name_to_item.keys())}")
            sys.exit(1)
        selected = [name_to_item[m] for m in models_arg]

    if not selected:
        print("No models selected.")
        sys.exit(1)

    print(f"\nSelected models: {[s['dir_name'] for s in selected]}")
    return selected


def _load_existing_prior_data_similarity(metrics_dir, expected_prior_names):
    """Load the newest matching prior_data_similarity_*.json cache if it exists."""
    if not os.path.isdir(metrics_dir):
        return None

    paths = [
        os.path.join(metrics_dir, name)
        for name in os.listdir(metrics_dir)
        if name.startswith("prior_data_similarity_") and name.endswith(".json")
    ]
    for path in sorted(paths, reverse=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        prior_names = payload.get("prior_names")
        data_matrix = payload.get("data_similarity_matrix")
        if not prior_names or data_matrix is None:
            continue

        if set(prior_names) != set(expected_prior_names):
            continue

        matrix = np.asarray(data_matrix, dtype=float)
        if matrix.shape != (len(prior_names), len(prior_names)):
            continue

        if prior_names != expected_prior_names:
            old_idx = {name: i for i, name in enumerate(prior_names)}
            order = [old_idx[name] for name in expected_prior_names]
            matrix = matrix[np.ix_(order, order)]
            prior_names = list(expected_prior_names)

        return {
            "path": path,
            "prior_names": prior_names,
            "metric_names": payload.get("metric_names", []),
            "data_similarity_matrix": matrix,
        }

    return None


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

            bucket_edges = torch.load(
                bucket_edges_path, map_location=device, weights_only=False
            )
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
        "--seed",
        type=int,
        default=config["training"]["seed"],
        help="Random seed (for decision boundary / regression toy plots)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Model folder name(s) to compare (non-interactive). "
            "Use 'all' for every discovered model, or list names e.g. 'tabpfn_mlp ticl_gp'. "
            "Leave empty for interactive selection."
        ),
    )
    parser.add_argument(
        "--skip_tabarena",
        action="store_true",
        help="Skip TabArena evaluation and heatmap generation.",
    )
    parser.add_argument(
        "--skip_data_similarity",
        action="store_true",
        help="Skip prior data-similarity plots after TabArena evaluation.",
    )
    parser.add_argument(
        "--tabarena_cache_dir",
        type=str,
        default=None,
        help="Directory to cache OpenML data for TabArena evaluation.",
    )

    args = parser.parse_args()

    # Defaults
    if args.models_dir is None:
        args.models_dir = os.path.join(
            os.path.dirname(__file__), args.problem_type, "results", "trained_models"
        )

    results_dir = os.path.join(os.path.dirname(__file__), args.problem_type, "results")
    data_dir = os.path.join(results_dir, "data")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.plot_output is None:
        os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
        args.plot_output = os.path.join(results_dir, "plots", f"comparison_{stamp}.png")
    if args.metrics_output is None:
        os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
        args.metrics_output = os.path.join(
            results_dir, "metrics", f"comparison_{stamp}_metrics.json"
        )

    # Discover and select trained models
    available = discover_trained_models(args.models_dir)
    available_priors = discover_h5_files(data_dir)
    if not available:
        print(f"No trained models found in {args.models_dir}")
        print("Run train_single_model.py first to train and save models.")
        sys.exit(1)

    if args.models is not None:
        selected = _select_models_noninteractive(available, args.models)
    else:
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

        run_records.append(
            {
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
            }
        )

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

    # --- TabArena heatmaps ---
    if not args.skip_tabarena:
        print(f"\n{'='*80}")
        print("TABARENA EVALUATION")
        print(f"{'='*80}\n")

        tabarena_metric = "ROC-AUC" if not is_regression else "R²"
        all_dataset_names: set[str] = set()
        per_prior_scores: dict[str, dict[str, float]] = (
            {}
        )  # prior_name -> {dataset -> score}

        for item in selected:
            meta = item["metadata"]
            prior_name = meta.get("prior_name", item["dir_name"])
            cache_path = os.path.join(item["dir"], "tabarena_results.json")

            # Check cache
            if os.path.isfile(cache_path):
                print(f"  {prior_name}: loading cached TabArena results")
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                per_prior_scores[prior_name] = cached
                all_dataset_names.update(cached.keys())
                continue

            # Load model and wrap in sklearn-like interface
            model_raw, dist = _load_model(
                item["dir"], device, is_regression=is_regression
            )

            if is_regression:
                if dist is None:
                    print(
                        f"  {prior_name}: SKIPPED (no bucket_edges.pth for regression)"
                    )
                    continue
                wrapped_model = NanoTabPFNRegressor(
                    model=model_raw, dist=dist, device=device
                )
            else:
                wrapped_model = NanoTabPFNClassifier(model=model_raw, device=device)

            print(f"  {prior_name}: evaluating on TabArena tasks...")
            predictions = get_openml_predictions(
                model=wrapped_model,
                tasks=TABARENA_TASKS,
                max_folds=5,
                max_n_samples=5_000,
                classification=not is_regression,
                cache_directory=args.tabarena_cache_dir,
            )

            # Compute per-dataset metric (average across folds)
            dataset_scores: dict[str, float] = {}
            for dataset_name, fold_records in predictions.items():
                fold_metrics = []
                for rec in fold_records:
                    y_true = rec["y_true"]
                    y_pred = rec["y_pred"]
                    y_proba = rec["y_proba"]
                    try:
                        if is_regression:
                            fold_metrics.append(r2_score(y_true, y_pred))
                        else:
                            if y_proba is not None:
                                fold_metrics.append(
                                    roc_auc_score(y_true, y_proba, multi_class="ovr")
                                )
                            else:
                                fold_metrics.append(roc_auc_score(y_true, y_pred))
                    except ValueError:
                        # e.g. single-class fold
                        continue

                if fold_metrics:
                    dataset_scores[dataset_name] = float(np.mean(fold_metrics))

            per_prior_scores[prior_name] = dataset_scores
            all_dataset_names.update(dataset_scores.keys())

            # Cache results
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(dataset_scores, f, indent=2)
            print(f"    cached to {cache_path}")

        # Build performance matrix
        all_dataset_names_sorted = sorted(all_dataset_names)
        prior_names_ordered = [
            meta.get("prior_name", item["dir_name"])
            for item, meta in [(item, item["metadata"]) for item in selected]
            if meta.get("prior_name", item["dir_name"]) in per_prior_scores
        ]

        if prior_names_ordered and all_dataset_names_sorted:
            perf_matrix = np.full(
                (len(prior_names_ordered), len(all_dataset_names_sorted)), np.nan
            )
            for i, pname in enumerate(prior_names_ordered):
                for j, dname in enumerate(all_dataset_names_sorted):
                    if dname in per_prior_scores.get(pname, {}):
                        perf_matrix[i, j] = per_prior_scores[pname][dname]

            # Save raw performance matrix as JSON
            perf_json_path = os.path.join(
                results_dir, "metrics", f"tabarena_performance_{stamp}.json"
            )
            os.makedirs(os.path.dirname(perf_json_path), exist_ok=True)
            with open(perf_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "prior_names": prior_names_ordered,
                        "dataset_names": all_dataset_names_sorted,
                        "metric": tabarena_metric,
                        "performance_matrix": perf_matrix.tolist(),
                    },
                    f,
                    indent=2,
                )
            print(f"\nSaved TabArena performance matrix to: {perf_json_path}")

            # Plot heatmaps
            heatmap_output = os.path.join(
                results_dir, "plots", f"tabarena_performance_{stamp}.png"
            )
            plot_tabarena_performance_heatmap(
                perf_matrix,
                prior_names_ordered,
                all_dataset_names_sorted,
                metric_name=tabarena_metric,
                output_path=heatmap_output,
            )

            normalized_heatmap_output = os.path.join(
                results_dir, "plots", f"tabarena_normalized_{stamp}.png"
            )
            plot_tabarena_normalized_heatmap(
                perf_matrix,
                prior_names_ordered,
                all_dataset_names_sorted,
                metric_name=tabarena_metric,
                output_path=normalized_heatmap_output,
            )

            if len(prior_names_ordered) >= 2:
                corr_output = os.path.join(
                    results_dir, "plots", f"prior_correlation_{stamp}.png"
                )
                plot_prior_correlation_heatmap(
                    perf_matrix,
                    prior_names_ordered,
                    output_path=corr_output,
                )

                if not args.skip_data_similarity:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    metrics_dir = os.path.join(results_dir, "metrics")
                    cached_similarity = _load_existing_prior_data_similarity(
                        metrics_dir, prior_names_ordered
                    )

                    if cached_similarity is not None:
                        aligned_prior_names = cached_similarity["prior_names"]
                        data_sim_matrix = cached_similarity["data_similarity_matrix"]
                        aligned_rows = [
                            perf_matrix[prior_names_ordered.index(pname)]
                            for pname in aligned_prior_names
                        ]
                        print(
                            "  prior_data_similarity_*.json already exists bro, "
                            f"using that: {cached_similarity['path']}"
                        )
                    else:
                        summary_vectors = []
                        summary_metric_names = None
                        aligned_prior_names = []
                        aligned_rows = []

                        for i, pname in enumerate(prior_names_ordered):
                            resolved_prior_path = available_priors.get(pname)

                            if resolved_prior_path is None:
                                print(
                                    f"  {pname}: SKIPPED data-similarity "
                                    f"(could not find generated prior in {data_dir})"
                                )
                                continue

                            print(
                                f"  {pname}: computing data-similarity summary",
                                flush=True,
                            )
                            analyzer = None
                            try:
                                if is_regression:
                                    analyzer = RegressionDataAnalyzer(
                                        resolved_prior_path
                                    )
                                else:
                                    analyzer = ClassificationDataAnalyzer(
                                        resolved_prior_path
                                    )
                                vec, metric_names = analyzer.prior_summary_vector()
                            finally:
                                del analyzer
                                gc.collect()

                            if summary_metric_names is None:
                                summary_metric_names = metric_names

                            aligned_prior_names.append(pname)
                            summary_vectors.append(vec)
                            aligned_rows.append(perf_matrix[i])

                        if len(summary_vectors) >= 2:
                            data_sim_matrix = compute_summary_similarity_matrix(
                                np.vstack(summary_vectors)
                            )
                            data_similarity_path = os.path.join(
                                metrics_dir,
                                f"prior_data_similarity_{stamp}.json",
                            )
                            os.makedirs(
                                os.path.dirname(data_similarity_path), exist_ok=True
                            )
                            with open(data_similarity_path, "w", encoding="utf-8") as f:
                                json.dump(
                                    {
                                        "prior_names": aligned_prior_names,
                                        "metric_names": summary_metric_names or [],
                                        "data_similarity_matrix": data_sim_matrix.tolist(),
                                    },
                                    f,
                                    indent=2,
                                )
                            print(
                                f"  Saved prior data similarity matrix to: {data_similarity_path}"
                            )

                    if len(aligned_prior_names) >= 2:

                        data_sim_output = os.path.join(
                            results_dir,
                            "plots",
                            f"prior_data_similarity_{stamp}.png",
                        )
                        plot_data_similarity_heatmap(
                            data_similarity_matrix=data_sim_matrix,
                            prior_names=aligned_prior_names,
                            output_path=data_sim_output,
                        )

                        perf_sim_matrix = compute_performance_similarity_matrix(
                            np.vstack(aligned_rows)
                        )

                        if perf_sim_matrix is None:
                            print(
                                "⚠️  Not enough complete TabArena datasets for "
                                "similarity-vs-similarity plot."
                            )
                        else:
                            sim_output = os.path.join(
                                results_dir,
                                "plots",
                                f"data_vs_performance_similarity_{stamp}.png",
                            )
                            stats_payload = plot_data_vs_performance_similarity(
                                data_similarity_matrix=data_sim_matrix,
                                performance_similarity_matrix=perf_sim_matrix,
                                prior_names=aligned_prior_names,
                                output_path=sim_output,
                            )
                            print(
                                "  Similarity readability plot generated for "
                                f"{stats_payload['num_pairs']} prior pairs"
                            )

                            sim_output = os.path.join(
                                results_dir,
                                "plots",
                                f"small_data_and_performance_similarity_{stamp}.png",
                            )
                            plot_small_data_and_performance_similarity_heatmaps(
                                data_similarity_matrix=data_sim_matrix,
                                performance_similarity_matrix=perf_sim_matrix,
                                prior_names=aligned_prior_names,
                                output_path=sim_output,
                            )

                            ranked_pairs = stats_payload.get("ranked_pairs", [])
                            if ranked_pairs:
                                print(
                                    "  Top pairs with largest data/performance mismatch:"
                                )
                                for row in ranked_pairs[:10]:
                                    print(
                                        f"    - {row['pair']}: "
                                        f"data={row['data_similarity']:.3f}, "
                                        f"perf={row['performance_similarity']:.3f}, "
                                        f"mismatch={row['gap']:.3f}"
                                    )

                                disagreement_path = os.path.join(
                                    results_dir,
                                    "metrics",
                                    f"data_vs_performance_disagreement_{stamp}.json",
                                )
                                os.makedirs(
                                    os.path.dirname(disagreement_path), exist_ok=True
                                )
                                with open(
                                    disagreement_path, "w", encoding="utf-8"
                                ) as f:
                                    json.dump(
                                        {
                                            "prior_names": aligned_prior_names,
                                            "pairs": ranked_pairs,
                                        },
                                        f,
                                        indent=2,
                                    )
                                print(
                                    f"  Saved mismatch ranking to: {disagreement_path}"
                                )
                            else:
                                print("  No valid prior-pair rows to report.")
                    else:
                        print(
                            "⚠️  Need at least 2 priors with valid prior files for "
                            "similarity scatter/table."
                        )
            else:
                print("⚠️  Need >= 2 priors for the correlation heatmap, skipping.")
        else:
            print("⚠️  No TabArena results to plot.")

    return {
        "problem_type": "regression" if is_regression else "classification",
        "metric_name": metric_name,
        "winner": winner,
        "runs": run_records,
    }


if __name__ == "__main__":
    results = main()
