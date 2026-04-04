"""Shared training and metrics utilities for nanoTabPFN experiments.

This module provides:
- train_model(): trains a single model on a prior
- _build_metrics_payload(): builds a structured metrics dict for JSON export
- _json_safe(): sanitises values for JSON serialisation
"""

import os
import time
from datetime import datetime, timezone

import torch
from pfns.bar_distribution import FullSupportBarDistribution
from torch import nn

from tfmplayground.evaluation import (
    TOY_TASKS_CLASSIFICATION,
    TOY_TASKS_REGRESSION,
)
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors import PriorDumpDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, make_global_bucket_edges

from tfmplayground.priors.experiments.classification.callback import (
    ClassificationTrackerCallback,
)
from tfmplayground.priors.experiments.regression.callback import (
    RegressionTrackerCallback,
)


def train_model(
    prior_path: str,
    model_name: str,
    epochs: int = 10,
    batch_size: int = 4,
    steps: int = 100,
    lr: float = 1e-4,
    device=None,
    eval_every: int = 1,
    tasks=None,
    n_buckets: int = 100,
    accumulate_gradients: int = 1,
    num_attention_heads: int = 4,
    embedding_size: int = 128,
    mlp_hidden_size: int = 512,
    num_layers: int = 6,
    checkpoint_base_dir: str = 'workdir',
    run_name: str | None = None,
):
    """
    Train a single nanoTabPFN model on the given prior.

    Args:
        prior_path: Path to the prior .h5 file
        model_name: Name for this model (used in logging)
        epochs: Target total number of training epochs
        batch_size: Batch size for training
        steps: Number of steps per epoch
        lr: Learning rate
        device: Device to train on
        eval_every: Evaluate toy tasks every N epochs
        tasks: OpenML task IDs for evaluation
        n_buckets: Number of buckets for regression bar distribution
        accumulate_gradients: Number of gradients to accumulate before updating weights
        checkpoint_base_dir:  Base directory under which run-specific checkpoint folders are stored
        run_name: Training run name used for checkpoint output

    Returns:
        Tuple of (trained_model, final_metric, callback, train_time,
                  inference_time, param_count, criterion)
        criterion is the FullSupportBarDistribution for regression, or
        CrossEntropyLoss for classification.
    """
    if device is None:
        device = get_default_device()

    if run_name is None:
        run_name = model_name.lower().replace(" ", "_")

    # load a saved checkpoint when continuing an existing prior-specific run.
    ckpt = None
    checkpoint_path = os.path.join(checkpoint_base_dir, run_name, "latest_checkpoint.pth")
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    print(f"\n{'='*80}")
    print(f"Training {model_name} on: {os.path.basename(prior_path)}")
    print(f"{'='*80}\n")

    # Load prior data
    prior = PriorDumpDataLoader(
        filename=prior_path,
        num_steps=steps,
        batch_size=batch_size,
        device=device,
        starting_index=steps * batch_size * (ckpt["epoch"] if ckpt else 0),
    )

    # Define problem type
    is_regression = prior.problem_type == "regression"

    # Prepare criterion
    if is_regression:
        # Compute bucket edges from the prior data (same as pretrain_regression.py)
        bucket_edges = make_global_bucket_edges(
            filename=prior_path,
            n_buckets=n_buckets,
            device=device,
        )
        criterion = FullSupportBarDistribution(bucket_edges).float().to(device)
        num_outputs = criterion.num_bars
    else:
        criterion = nn.CrossEntropyLoss()
        num_outputs = prior.max_num_classes if prior.max_num_classes else 1


    model = NanoTabPFNModel(
        num_attention_heads=num_attention_heads,
        embedding_size=embedding_size,
        mlp_hidden_size=mlp_hidden_size,
        num_layers=num_layers,
        num_outputs=num_outputs,
    )

    if ckpt:
        model.load_state_dict(ckpt["model"])

    if ckpt and ckpt["epoch"] >= epochs:
        raise ValueError(
            f"Checkpoint is already at epoch {ckpt['epoch']}, but --epochs={epochs}. "
            "Use a larger --epochs value to continue training."
        )

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())

    # Define callback based on problem type
    if is_regression:
        use_tasks = tasks if tasks is not None else TOY_TASKS_REGRESSION
        callback = RegressionTrackerCallback(
            use_tasks, model_name, eval_every=eval_every
        )
    else:
        use_tasks = tasks if tasks is not None else TOY_TASKS_CLASSIFICATION
        callback = ClassificationTrackerCallback(
            use_tasks, model_name, eval_every=eval_every
        )

    # Train the model and track time
    train_start = time.time()
    trained_model, _ = train(
        model=model,
        prior=prior,
        criterion=criterion,
        epochs=epochs,
        accumulate_gradients=accumulate_gradients,
        lr=lr,
        device=device,
        callbacks=[callback],
        ckpt=ckpt,
        run_name=run_name,
        checkpoint_base_dir=checkpoint_base_dir,
    )
    train_time = time.time() - train_start

    # Measure inference time on a sample batch
    trained_model.eval()
    with torch.no_grad():
        sample_batch = next(iter(prior))
        # Extract single_eval_pos as int (may be tensor or int)
        single_eval_pos = sample_batch["single_eval_pos"]
        if isinstance(single_eval_pos, torch.Tensor):
            single_eval_pos = single_eval_pos.item()

        inference_start = time.time()
        for _ in range(100):  # Average over 100 runs
            _ = trained_model(
                (sample_batch["x"], sample_batch["y"]), single_eval_pos=single_eval_pos
            )
        inference_time = (time.time() - inference_start) / 100

    return (
        trained_model,
        callback.final_rmse if is_regression else callback.final_roc_auc,
        callback,
        train_time,
        inference_time,
        param_count,
        criterion,
    )


def _json_safe(value):
    """Recursively convert a value into a JSON-safe representation."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        return _json_safe(value.item())
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
    return value


def _last_non_none(values):
    """Return the last non-None element from a list."""
    for v in reversed(values):
        if v is not None:
            return v
    return None


def _build_metrics_payload(run_records, metric_name: str, is_regression: bool):
    """Build a structured metrics dict suitable for JSON export."""
    sorted_runs = sorted(
        run_records, key=lambda r: r["metric"], reverse=not is_regression
    )
    winner_record = sorted_runs[0] if sorted_runs else None

    models = []
    for r in run_records:
        metric_history = r["metric_history"]
        valid_metric_history = [
            (idx + 1, v) for idx, v in enumerate(metric_history) if v is not None
        ]

        if valid_metric_history:
            if is_regression:
                best_epoch, best_metric = min(valid_metric_history, key=lambda x: x[1])
            else:
                best_epoch, best_metric = max(valid_metric_history, key=lambda x: x[1])
            final_metric = valid_metric_history[-1][1]
        else:
            best_epoch, best_metric, final_metric = None, None, None

        if is_regression:
            final_task_scores = {}
            for dataset_name, per_epoch_folds in r["per_task_scores"].items():
                if not per_epoch_folds:
                    final_task_scores[dataset_name] = None
                    continue
                last_fold_values = per_epoch_folds[-1]
                if not last_fold_values:
                    final_task_scores[dataset_name] = None
                else:
                    final_task_scores[dataset_name] = sum(last_fold_values) / len(
                        last_fold_values
                    )
        else:
            #TODO: potential bug pointed out by the agent, check this
            final_task_scores = {
                dataset_name: _last_non_none(values)
                for dataset_name, values in r["per_task_scores"].items()
            }

        model_payload = {
            "model_index": r["index"],
            "model_name": r["model_name"],
            "prior": r["prior"],
            "prior_name": r["prior_name"],
            "metric_name": metric_name,
            "epochs": list(range(1, len(r["loss_history"]) + 1)),
            "epoch_times": r["callback"].epoch_times,
            "losses": r["loss_history"],
            "metric_history": metric_history,
            "task_scores": r["per_task_scores"],
            "final_task_scores": final_task_scores,
            "train_time": r["train_time"],
            "inference_time": r["inference_time"],
            "param_count": r["param_count"],
            "final_metric": final_metric,
            "final_loss": _last_non_none(r["loss_history"]),
            "best_epoch": best_epoch,
            "best_metric": best_metric,
        }
        models.append(model_payload)

    return {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        },
        "models": models,
        "summary": {
            "winner_index": winner_record["index"] if winner_record else None,
            "winner_prior": winner_record["prior"] if winner_record else None,
            "best_metric": winner_record["metric"] if winner_record else None,
            "num_models": len(run_records),
        },
    }
