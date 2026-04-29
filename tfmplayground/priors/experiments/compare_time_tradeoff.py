"""Compare prior generation cost against downstream performance.

This script joins:
- prior generation metrics from H5 attributes (wall_seconds, ...)
- trained model metrics from metadata.json (train_time, final_metric, ...)

It then produces generation-cost plots and a joined metrics JSON/CSV.

Usage:
    python tfmplayground/priors/experiments/compare_time_tradeoff.py --problem_type classification
    python tfmplayground/priors/experiments/compare_time_tradeoff.py --problem_type classification --priors tabforest_cuts tabforest_forest tabforest_neighbor
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from tfmplayground.priors.experiments.utils.general import (
    discover_trained_models,
    load_config,
)
from tfmplayground.priors.experiments.utils.visualization import _get_n_colors


def _select_models_noninteractive(available, priors_arg):
    """Resolve --priors CLI argument to model items by prior_name."""
    if priors_arg == ["all"]:
        selected = available
    else:
        prior_to_item = {
            item["metadata"].get("prior_name", item["dir_name"]): item
            for item in available
        }
        unknown = [p for p in priors_arg if p not in prior_to_item]
        if unknown:
            print(f"ERROR: Unknown prior(s): {unknown}")
            print(f"Available: {list(prior_to_item.keys())}")
            raise SystemExit(1)
        selected = [prior_to_item[p] for p in priors_arg]

    if not selected:
        print("No priors selected.")
        raise SystemExit(1)

    print(f"\nSelected priors: {[s['metadata'].get('prior_name', s['dir_name']) for s in selected]}")
    return selected


def _extract_prior_name_from_h5_filename(path: Path) -> str | None:
    """Extract prior name from filenames like prior_<name>_250000x1_200x10.h5."""
    stem = path.stem
    match = re.match(r"^prior_([a-zA-Z_]+)", stem)
    if not match:
        return None
    return match.group(1).rstrip("_")


def _discover_h5_generation_metrics(data_dir: str):
    """Discover generation metrics from H5 attrs keyed by prior_name."""
    data_dir_path = Path(data_dir)
    if not data_dir_path.is_dir():
        return {}

    discovered = {}
    for file_path in sorted(data_dir_path.glob("prior_*.h5")):
        prior_name = _extract_prior_name_from_h5_filename(file_path)
        if not prior_name:
            continue

        attrs = {
            "wall_seconds": None,
            "cpu_seconds": None,
            "peak_memory_bytes": None,
            "throughput_samples_per_cpu_sec": None,
            "total_samples": None,
        }

        try:
            with h5py.File(file_path, "r") as f:
                for key in attrs:
                    value = f.attrs.get(key)
                    attrs[key] = value.item() if hasattr(value, "item") else value
        except OSError:
            continue

        discovered[prior_name] = {
            "h5_path": str(file_path),
            **attrs,
        }

    return discovered


def _as_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_tabarena_average(model_dir: str):
    """Load cached TabArena per-dataset scores and return their mean."""
    cache_path = os.path.join(model_dir, "tabarena_results.json")
    if not os.path.isfile(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None

    values = []
    for score in data.values():
        val = _as_float_or_none(score)
        if val is not None:
            values.append(val)

    if not values:
        return None

    return float(np.mean(values))


def _build_joined_records(selected_models, h5_metrics):
    """Join model metadata and H5 attrs by prior_name."""
    rows = []
    for item in selected_models:
        meta = item["metadata"]
        prior_name = meta.get("prior_name", item["dir_name"])
        gen = h5_metrics.get(prior_name, {})

        generation_time = _as_float_or_none(gen.get("wall_seconds"))
        train_time = _as_float_or_none(meta.get("train_time"))
        training_final_metric = _as_float_or_none(meta.get("final_metric"))
        tabarena_avg_metric = _load_tabarena_average(item["dir"])
        # Prefer benchmark average for cost-benefit plots; fall back to training metric.
        final_metric = (
            tabarena_avg_metric
            if tabarena_avg_metric is not None
            else training_final_metric
        )

        problem_type = meta.get("problem_type")
        if tabarena_avg_metric is not None:
            metric_name = "ROC-AUC (TabArena avg)" if problem_type == "classification" else "R² (TabArena avg)"
        else:
            metric_name = meta.get("metric_name", "Metric")

        total_time = None
        if generation_time is not None and train_time is not None:
            total_time = generation_time + train_time

        rows.append(
            {
                "prior_name": prior_name,
                "model_dir": item["dir"],
                "problem_type": problem_type,
                "metric_name": metric_name,
                "final_metric": final_metric,
                "training_final_metric": training_final_metric,
                "tabarena_avg_metric": tabarena_avg_metric,
                "train_time": train_time,
                "generation_wall_seconds": generation_time,
                "generation_cpu_seconds": _as_float_or_none(gen.get("cpu_seconds")),
                "generation_peak_memory_bytes": _as_float_or_none(gen.get("peak_memory_bytes")),
                "generation_total_samples": _as_float_or_none(gen.get("total_samples")),
                "generation_h5_path": gen.get("h5_path"),
                "total_time": total_time,
            }
        )

    return rows


def _add_improvement_columns(rows, baseline_prior, is_regression):
    """Compute metric improvement vs baseline and efficiency."""
    baseline = next((r for r in rows if r["prior_name"] == baseline_prior), None)
    if baseline is None or baseline.get("final_metric") is None:
        return None

    baseline_metric = baseline["final_metric"]
    for row in rows:
        metric = row.get("final_metric")
        if metric is None:
            row["improvement_vs_baseline"] = None
            row["improvement_per_second"] = None
            continue

        if is_regression:
            improvement = baseline_metric - metric
        else:
            improvement = metric - baseline_metric

        row["improvement_vs_baseline"] = improvement
        total_time = row.get("total_time")
        if total_time is None or total_time <= 0:
            row["improvement_per_second"] = None
        else:
            row["improvement_per_second"] = improvement / total_time

    return baseline_metric


def _save_csv(rows, output_path):
    if not rows:
        return

    fieldnames = [
        "prior_name",
        "problem_type",
        "metric_name",
        "final_metric",
        "tabarena_avg_metric",
        "training_final_metric",
        "generation_wall_seconds",
        "generation_cpu_seconds",
        "generation_peak_memory_bytes",
        "generation_total_samples",
        "train_time",
        "total_time",
        "improvement_vs_baseline",
        "improvement_per_second",
        "generation_h5_path",
        "model_dir",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _plot_stacked_cost(rows, output_path):
    plot_rows = [
        r
        for r in rows
        if r.get("generation_wall_seconds") is not None
    ]
    if not plot_rows:
        print("Skipping stacked cost plot (no rows with generation time).")
        return

    names = [r["prior_name"] for r in plot_rows]
    gen = np.array([r["generation_wall_seconds"] for r in plot_rows], dtype=float)
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(names)), 5))
    ax.bar(x, gen, label="Generation time", color="#5B8FF9")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Time (s)")
    ax.set_title("Per-prior generation cost")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved stacked cost plot: {output_path}")


def _plot_performance_vs_cost(rows, output_path):
    plot_rows = [
        r
        for r in rows
        if r.get("generation_wall_seconds") is not None and r.get("final_metric") is not None
    ]
    if not plot_rows:
        print("Skipping performance-vs-cost plot (no rows with generation time and final_metric).")
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = _get_n_colors(len(plot_rows))

    for idx, row in enumerate(plot_rows):
        ax.scatter(
            row["generation_wall_seconds"],
            row["final_metric"],
            s=90,
            color=colors[idx],
            label=row["prior_name"],
            edgecolors="white",
            linewidths=0.7,
        )

    ax.set_xlabel("Generation time (s)")
    metric_name = plot_rows[0].get("metric_name", "Metric")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs generation cost")
    ax.grid(True, alpha=0.3)
    ax.legend(
        title="Prior",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
        frameon=True,
        fontsize=9,
        title_fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 0.78, 1))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved performance-vs-cost plot: {output_path}")


def _filter_real_priors(rows, exclude_real_priors):
    """Optionally drop real_* priors from a time-tradeoff plot."""
    if not exclude_real_priors:
        return rows
    return [row for row in rows if not row["prior_name"].startswith("real_")]


def _plot_improvement_vs_cost(rows, baseline_prior, baseline_metric, output_path):
    plot_rows = [
        r
        for r in rows
        if r.get("generation_wall_seconds") is not None and r.get("improvement_vs_baseline") is not None
    ]
    if not plot_rows:
        print("Skipping improvement-vs-cost plot (missing baseline improvements or generation time).")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for row in plot_rows:
        is_baseline = row["prior_name"] == baseline_prior
        marker = "*" if is_baseline else "o"
        size = 140 if is_baseline else 70
        ax.scatter(row["generation_wall_seconds"], row["improvement_vs_baseline"], s=size, marker=marker)
        ax.annotate(
            row["prior_name"],
            (row["generation_wall_seconds"], row["improvement_vs_baseline"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xlabel("Generation time (s)")
    ax.set_ylabel(f"Improvement vs baseline ({baseline_prior})")
    ax.set_title(f"Metric gain vs generation cost (baseline={baseline_prior}, metric={baseline_metric:.4f})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved improvement-vs-cost plot: {output_path}")


def main():
    config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))

    parser = argparse.ArgumentParser(
        description="Compare prior generation time tradeoffs",
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        choices=["classification", "regression"],
        required=True,
        help="Problem type (determines which results folders are used).",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=None,
        help="Directory containing trained model folders (default: <problem_type>/results/trained_models/).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing generated prior H5 files (default: <problem_type>/results/data/).",
    )
    parser.add_argument(
        "--priors",
        type=str,
        nargs="+",
        default=["all"],
        help="Priors to include. Use 'all' for every discovered trained prior.",
    )
    parser.add_argument(
        "--baseline_prior",
        type=str,
        default=None,
        help="Baseline prior name for improvement-vs-cost plot. Defaults to first selected prior.",
    )
    parser.add_argument(
        "--allow_missing_generation_time",
        action="store_true",
        help="Allow rows without generation wall_seconds in joined outputs.",
    )
    parser.add_argument(
        "--exclude_real_priors",
        action="store_true",
        help="Hide priors whose names start with real_ from the time-tradeoff plots.",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=None,
        help="Directory to save plots. Defaults to <problem_type>/results/plots/.",
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default=None,
        help="Directory to save joined metrics JSON/CSV. Defaults to <problem_type>/results/metrics/.",
    )
    args = parser.parse_args()

    base_results = os.path.join(os.path.dirname(__file__), args.problem_type, "results")
    if args.models_dir is None:
        args.models_dir = os.path.join(base_results, "trained_models")
    if args.data_dir is None:
        args.data_dir = os.path.join(base_results, "data")
    if args.plots_dir is None:
        args.plots_dir = os.path.join(base_results, "plots")
    if args.metrics_dir is None:
        args.metrics_dir = os.path.join(base_results, "metrics")

    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)

    available = discover_trained_models(args.models_dir)
    if not available:
        print(f"No trained models found in {args.models_dir}")
        raise SystemExit(1)

    selected = _select_models_noninteractive(available, args.priors)
    h5_metrics = _discover_h5_generation_metrics(args.data_dir)
    rows = _build_joined_records(selected, h5_metrics)

    if not args.allow_missing_generation_time:
        rows = [r for r in rows if r.get("generation_wall_seconds") is not None]
        if not rows:
            print("No rows remain after filtering for generation wall_seconds.")
            raise SystemExit(1)

    is_regression = args.problem_type == "regression"
    baseline_prior = args.baseline_prior or rows[0]["prior_name"]
    baseline_metric = _add_improvement_columns(rows, baseline_prior, is_regression)
    if baseline_metric is None:
        print(
            f"Could not compute baseline improvements: missing baseline prior or baseline metric for '{baseline_prior}'."
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    joined_json = os.path.join(args.metrics_dir, f"time_tradeoff_{stamp}.json")
    joined_csv = os.path.join(args.metrics_dir, f"time_tradeoff_{stamp}.csv")

    payload = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "problem_type": args.problem_type,
            "models_dir": os.path.abspath(args.models_dir),
            "data_dir": os.path.abspath(args.data_dir),
            "allow_missing_generation_time": args.allow_missing_generation_time,
            "baseline_prior": baseline_prior,
            "baseline_metric": baseline_metric,
            "num_rows": len(rows),
        },
        "rows": rows,
    }

    with open(joined_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _save_csv(rows, joined_csv)
    print(f"Saved joined metrics JSON: {joined_json}")
    print(f"Saved joined metrics CSV : {joined_csv}")

    plot_rows = _filter_real_priors(rows, args.exclude_real_priors)
    if args.exclude_real_priors:
        excluded = sorted(
            row["prior_name"] for row in rows if row["prior_name"].startswith("real_")
        )
        print(f"Excluding real priors from scatter plots: {excluded}")

    _plot_stacked_cost(rows, os.path.join(args.plots_dir, f"time_tradeoff_stacked_{stamp}.png"))
    _plot_performance_vs_cost(plot_rows, os.path.join(args.plots_dir, f"time_tradeoff_perf_vs_cost_{stamp}.png"))
    if baseline_metric is not None:
        _plot_improvement_vs_cost(
            plot_rows,
            baseline_prior=baseline_prior,
            baseline_metric=baseline_metric,
            output_path=os.path.join(args.plots_dir, f"time_tradeoff_gain_vs_cost_{stamp}.png"),
        )


if __name__ == "__main__":
    main()
