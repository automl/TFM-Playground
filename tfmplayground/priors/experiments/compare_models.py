import argparse
import os
import sys
import time

import requests
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.metrics import roc_auc_score, r2_score
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.evaluation import (TOY_TASKS_CLASSIFICATION,
                                      TOY_TASKS_REGRESSION,
                                      get_openml_predictions)
from tfmplayground.interface import NanoTabPFNClassifier, NanoTabPFNRegressor
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors import PriorDumpDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed
from visualization_utils import (
    plot_comparison_multi,
    plot_all_decision_boundaries,
    plot_all_regression_predictions,
)


class ClassificationTrackerCallback(ConsoleLoggerCallback):
    """Callback that tracks accuracy on toy tasks and stores the final accuracy and loss history."""

    def __init__(self, tasks, model_name="Model", eval_every: int = 1):
        self.tasks = tasks
        self.model_name = model_name
        self.eval_every = max(1, int(eval_every))
        self.final_accuracy = 0.0
        self.device = get_default_device()
        self.loss_history = []
        self.accuracy_history = []  # may contain None for skipped epochs
        self.epoch_history = []

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        # Always track loss per epoch
        self.epoch_history.append(epoch)
        self.loss_history.append(loss)

        # Optionally skip expensive evaluation
        if (epoch % self.eval_every) != 0:
            self.accuracy_history.append(None)
            print(
                f"[{self.model_name}] epoch {epoch:5d} | time {epoch_time:5.2f}s | "
                f"mean loss {loss:5.2f} | eval skipped (every {self.eval_every})",
                flush=True,
            )
            return

        classifier = NanoTabPFNClassifier(model, self.device)
        predictions = get_openml_predictions(model=classifier, tasks=self.tasks)
        scores = []
        for dataset_name, (y_true, y_pred, y_proba) in predictions.items():
            scores.append(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        avg_score = sum(scores) / len(scores) if len(scores) else float("nan")
        self.final_accuracy = avg_score
        self.accuracy_history.append(avg_score)

        print(
            f"[{self.model_name}] epoch {epoch:5d} | time {epoch_time:5.2f}s | "
            f"mean loss {loss:5.2f} | avg accuracy {avg_score:.3f}",
            flush=True,
        )


class RegressionTrackerCallback(ConsoleLoggerCallback):
    """Callback that tracks R2 on toy tasks and stores the final R2 and loss history."""

    def __init__(self, tasks, model_name="Model", eval_every: int = 1):
        self.tasks = tasks
        self.model_name = model_name
        self.eval_every = max(1, int(eval_every))
        self.final_score = 0.0
        self.device = get_default_device()
        self.loss_history = []
        self.score_history = []  # may contain None for skipped epochs
        self.epoch_history = []

    def on_epoch_end(
        self, epoch: int, epoch_time: float, loss: float, model, dist=None, **kwargs
    ):
        # Always track loss per epoch
        self.epoch_history.append(epoch)
        self.loss_history.append(loss)

        # Optionally skip expensive evaluation
        if (epoch % self.eval_every) != 0:
            self.score_history.append(None)
            print(
                f"[{self.model_name}] epoch {epoch:5d} | time {epoch_time:5.2f}s | "
                f"mean loss {loss:5.2f} | eval skipped (every {self.eval_every})",
                flush=True,
            )
            return

        # Use the full NanoTabPFNRegressor which handles the distribution
        regressor = NanoTabPFNRegressor(model=model, dist=dist, device=self.device)
        predictions = get_openml_predictions(
            model=regressor, tasks=self.tasks, classification=False
        )
        scores = []
        for dataset_name, (y_true, y_pred, _) in predictions.items():
            scores.append(r2_score(y_true, y_pred))
        avg_score = sum(scores) / len(scores) if len(scores) else float("nan")
        self.final_score = avg_score
        self.score_history.append(avg_score)

        print(
            f"[{self.model_name}] epoch {epoch:5d} | time {epoch_time:5.2f}s | "
            f"mean loss {loss:5.2f} | avg R2 {avg_score:.3f}",
            flush=True,
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
    buckets_path: str = "checkpoints/nanotabpfn_regressor_buckets.pth",
):
    """
    Train a single nanoTabPFN model on the given prior.

    Args:
        prior_path: Path to the prior .h5 file
        model_name: Name for this model (used in logging)
        epochs: Number of training epochs
        batch_size: Batch size for training
        steps: Number of steps per epoch
        lr: Learning rate
        device: Device to train on

    Returns:
        Tuple of (trained_model, final_accuracy, callback, train_time, inference_time, param_count)
    """
    if device is None:
        device = get_default_device()

    print(f"\n{'='*80}")
    print(f"Training {model_name} on: {os.path.basename(prior_path)}")
    print(f"{'='*80}\n")

    # Load prior data
    prior = PriorDumpDataLoader(
        filename=prior_path,
        num_steps=steps,
        batch_size=batch_size,
        device=device,
        starting_index=0,
    )

    # Define problem type
    is_regression = prior.problem_type == "regression"

    # Prepare criterion for regression (FullSupportBarDistribution)
    if is_regression:
        if not os.path.isfile(buckets_path):
            print(f"Downloading bucket edges to {buckets_path}...")
            os.makedirs(os.path.dirname(buckets_path), exist_ok=True)
            response = requests.get(
                "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor_buckets.pth"
            )
            with open(buckets_path, "wb") as f:
                f.write(response.content)

        bucket_edges = torch.load(buckets_path, map_location=device)
        criterion = FullSupportBarDistribution(bucket_edges).float().to(device)
        num_outputs = criterion.num_bars
    else:
        criterion = nn.CrossEntropyLoss()
        num_outputs = prior.max_num_classes if prior.max_num_classes else 1

    # Create model
    model = NanoTabPFNModel(
        num_attention_heads=6,
        embedding_size=192,
        mlp_hidden_size=768,
        num_layers=6,
        num_outputs=num_outputs,
    )

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())

    # Define callback based on problem type
    if is_regression:
        use_tasks = tasks if tasks is not None else TOY_TASKS_REGRESSION
        callback = RegressionTrackerCallback(use_tasks, model_name, eval_every=eval_every)
    else:
        use_tasks = tasks if tasks is not None else TOY_TASKS_CLASSIFICATION
        callback = ClassificationTrackerCallback(use_tasks, model_name, eval_every=eval_every)

    # Train the model and track time
    train_start = time.time()
    trained_model, _ = train(
        model=model,
        prior=prior,
        criterion=criterion,
        epochs=epochs,
        accumulate_gradients=1,
        lr=lr,
        device=device,
        callbacks=[callback],
        run_name=f"compare_{model_name.lower().replace(' ', '_')}",
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
        callback.final_score if is_regression else callback.final_accuracy,
        callback,
        train_time,
        inference_time,
        param_count,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare nanoTabPFN models on multiple priors"
    )
    parser.add_argument(
        "--prior",
        action="append",
        default=[],
        help="Path to a prior .h5 file. Repeat this flag to provide multiple priors.",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train each model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--steps", type=int, default=5, help="Number of steps per epoch"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--seed", type=int, default=2402, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--plot_output",
        type=str,
        default="comparison_plot.png",
        help="Path to save the comparison plot",
    )

    parser.add_argument(
        "--buckets_path",
        type=str,
        default="checkpoints/nanotabpfn_regressor_buckets.pth",
        help="Path to bucket edges for regression",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="Evaluate toy tasks every N epochs (default: 1)",
    )
    parser.add_argument(
        "--toy_tasks_subset",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of OpenML task IDs to evaluate on (space-separated)",
    )

    args = parser.parse_args()

    priors = args.prior
    if priors is None or len(priors) < 2:
        raise ValueError("Please provide at least two --prior arguments.")

    # Set random seed
    set_randomness_seed(args.seed)

    # Get device
    device = get_default_device()
    print(f"Using device: {device}\n")

    # Determine which toy tasks to use
    subset = set(args.toy_tasks_subset) if args.toy_tasks_subset else None
    cls_tasks = [t for t in TOY_TASKS_CLASSIFICATION if (subset is None or t in subset)]
    reg_tasks = [t for t in TOY_TASKS_REGRESSION if (subset is None or t in subset)]

    run_records = []
    callbacks = []
    prior_names = []

    expected_problem_type = None
    is_regression = None

    for idx, prior_path in enumerate(priors, start=1):
        # Probe prior to determine problem type and enforce consistency
        prior_probe = PriorDumpDataLoader(
            filename=prior_path,
            num_steps=1,
            batch_size=1,
            device=device,
            starting_index=0,
        )
        this_problem_type = prior_probe.problem_type
        this_is_regression = this_problem_type == "regression"

        if expected_problem_type is None:
            expected_problem_type = this_problem_type
            is_regression = this_is_regression
        elif this_problem_type != expected_problem_type:
            raise ValueError(
                f"Mixed problem types are not allowed. Expected {expected_problem_type} but got {this_problem_type} for {prior_path}."
            )

        use_tasks = reg_tasks if is_regression else cls_tasks

        # Reset seed for fair comparison
        set_randomness_seed(args.seed)

        model_name = f"Model {idx}"
        trained_model, metric, callback, train_time, inference_time, param_count = train_model(
            prior_path=prior_path,
            model_name=model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
            device=device,
            eval_every=args.eval_every,
            tasks=use_tasks,
            buckets_path=args.buckets_path,
        )

        pname = os.path.basename(prior_path).replace(".h5", "").replace("prior_", "")
        prior_names.append(pname)
        callbacks.append(callback)

        run_records.append(
            {
                "index": idx,
                "model_name": model_name,
                "prior": prior_path,
                "prior_name": pname,
                "metric": metric,
                "loss_history": callback.loss_history,
                "metric_history": (
                    callback.score_history if is_regression else callback.accuracy_history
                ),
                "train_time": train_time,
                "inference_time": inference_time,
                "param_count": param_count,
                "model": trained_model,
            }
        )

    metric_name = "R2 Score" if is_regression else "Accuracy"

    print(f"\n{'='*80}")
    print("FINAL COMPARISON RESULTS")
    print(f"{'='*80}\n")

    sorted_runs = sorted(run_records, key=lambda r: r["metric"], reverse=True)
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

    plot_comparison_multi(
        callbacks=callbacks,
        prior_names=prior_names,
        save_path=args.plot_output,
        metric_name=metric_name,
    )

    # Plot decision boundaries for classification tasks only
    if not is_regression:
        decision_boundary_output = args.plot_output.replace(".png", "_decision_boundaries.png")
        plot_all_decision_boundaries(
            run_records,
            datasets=["moons", "circles"],
            n_samples=200,
            noise=0.2,
            seed=args.seed,
            output_path=decision_boundary_output,
        )
    else:
        # Plot regression predictions for regression tasks
        regression_output = args.plot_output.replace(".png", "_regression_predictions.png")
        plot_all_regression_predictions(
            run_records,
            datasets=["sinusoidal", "linear", "step"],
            n_samples=100,
            noise=0.1,
            seed=args.seed,
            output_path=regression_output,
        )

    return {
        "problem_type": expected_problem_type,
        "metric_name": metric_name,
        "winner": winner,
        "runs": run_records,
    }


if __name__ == "__main__":
    results = main()
