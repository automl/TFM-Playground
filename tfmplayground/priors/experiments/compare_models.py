#!/usr/bin/env python3
"""
Script to train and compare two nanoTabPFN models on different classification priors.

This script:
1. Loads two classification prior files
2. Trains two nanoTabPFN models for 10 epochs each
3. Evaluates both models on toy classification tasks
4. Compares their final accuracy
"""

import argparse
import time
import torch
from torch import nn
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import numpy as np
import requests
from pfns.bar_distribution import FullSupportBarDistribution
import os
import sys

# Add parent directory to path to import from tfmplayground
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.evaluation import get_openml_predictions, TOY_TASKS_CLASSIFICATION, TOY_TASKS_REGRESSION
from tfmplayground.interface import NanoTabPFNClassifier, NanoTabPFNRegressor
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors import PriorDumpDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed


class ClassificationTrackerCallback(ConsoleLoggerCallback):
    """Callback that tracks accuracy on toy tasks and stores the final accuracy and loss history."""

    def __init__(self, tasks, model_name="Model"):
        self.tasks = tasks
        self.model_name = model_name
        self.final_accuracy = 0.0
        self.device = get_default_device()
        self.loss_history = []
        self.accuracy_history = []
        self.epoch_history = []

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        classifier = NanoTabPFNClassifier(model, self.device)
        predictions = get_openml_predictions(model=classifier, tasks=self.tasks)
        scores = []
        for dataset_name, (y_true, y_pred, y_proba) in predictions.items():
            scores.append(accuracy_score(y_true, y_pred))
        avg_score = sum(scores) / len(scores)
        self.final_accuracy = avg_score

        # Track history
        self.epoch_history.append(epoch)
        self.loss_history.append(loss)
        self.accuracy_history.append(avg_score)

        print(
            f"[{self.model_name}] epoch {epoch:5d} | time {epoch_time:5.2f}s | "
            f"mean loss {loss:5.2f} | avg accuracy {avg_score:.3f}",
            flush=True,
        )




class RegressionTrackerCallback(ConsoleLoggerCallback):
    """Callback that tracks R2 on toy tasks and stores the final R2 and loss history."""

    def __init__(self, tasks, model_name="Model"):
        self.tasks = tasks
        self.model_name = model_name
        self.final_score = 0.0
        self.device = get_default_device()
        self.loss_history = []
        self.score_history = []
        self.epoch_history = []

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, dist=None, **kwargs):
        # Use the full NanoTabPFNRegressor which handles the distribution
        regressor = NanoTabPFNRegressor(model=model, dist=dist, device=self.device)
        predictions = get_openml_predictions(
            model=regressor, tasks=self.tasks, classification=False
        )
        scores = []
        for dataset_name, (y_true, y_pred, _) in predictions.items():
            scores.append(r2_score(y_true, y_pred))
        avg_score = sum(scores) / len(scores)
        self.final_score = avg_score

        # Track history
        self.epoch_history.append(epoch)
        self.loss_history.append(loss)
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
        # criterion is already set to FullSupportBarDistribution
        callback = RegressionTrackerCallback(TOY_TASKS_REGRESSION, model_name)
    else:
        # criterion is already set to CrossEntropyLoss
        callback = ClassificationTrackerCallback(TOY_TASKS_CLASSIFICATION, model_name)

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


def plot_comparison(
    callback1, callback2, prior1_name, prior2_name, save_path="comparison_plot.png", metric_name="Accuracy"
):
    """
    Create comparison plots for loss and accuracy curves.

    Args:
        callback1: First model's callback with history
        callback2: Second model's callback with history
        prior1_name: Name of first prior (for legend)
        prior2_name: Name of second prior (for legend)
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss curves
    ax1.plot(
        callback1.epoch_history,
        callback1.loss_history,
        "b-o",
        label=f"Model 1: {prior1_name}",
        linewidth=2,
        markersize=6,
    )
    ax1.plot(
        callback2.epoch_history,
        callback2.loss_history,
        "r-s",
        label=f"Model 2: {prior2_name}",
        linewidth=2,
        markersize=6,
    )
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training Loss Comparison", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot accuracy curves
    # Plot accuracy/score curves
    metric_history1 = callback1.score_history if hasattr(callback1, "score_history") else callback1.accuracy_history
    metric_history2 = callback2.score_history if hasattr(callback2, "score_history") else callback2.accuracy_history
    
    ax2.plot(
        callback1.epoch_history,
        metric_history1,
        "b-o",
        label=f"Model 1: {prior1_name}",
        linewidth=2,
        markersize=6,
    )
    ax2.plot(
        callback2.epoch_history,
        metric_history2,
        "r-s",
        label=f"Model 2: {prior2_name}",
        linewidth=2,
        markersize=6,
    )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel(metric_name, fontsize=12)
    ax2.set_title(f"Validation {metric_name} Comparison", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Comparison plot saved to: {save_path}")

    # Also try to display if in interactive environment
    try:
        plt.show()
    except:
        pass

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare two nanoTabPFN models on different classification priors"
    )
    parser.add_argument(
        "--prior1",
        type=str,
        default="tfmplayground/priors/experiments/results/data/prior_ticl_classification_adapter_1x8_1024x100.h5",
        help="Path to first prior file",
    )
    parser.add_argument(
        "--prior2",
        type=str,
        default="tfmplayground/priors/experiments/results/data/prior_tabicl_1x8_1024x100.h5",
        help="Path to second prior file",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train each model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--steps", type=int, default=1, help="Number of steps per epoch"
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
    
    args = parser.parse_args()

    # Set random seed
    set_randomness_seed(args.seed)

    # Get device
    device = get_default_device()
    print(f"Using device: {device}\n")

    # Train first model
    model1, accuracy1, callback1, train_time1, inference_time1, param_count1 = (
        train_model(
            prior_path=args.prior1,
            model_name="Model 1",
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
            device=device,
            buckets_path=args.buckets_path,
        )
    )

    # Reset seed for fair comparison
    set_randomness_seed(args.seed)

    # Train second model
    model2, accuracy2, callback2, train_time2, inference_time2, param_count2 = (
        train_model(
            prior_path=args.prior2,
            model_name="Model 2",
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
            device=device,
            buckets_path=args.buckets_path,
        )
    )

    # Print comparison results
    print(f"\n{'='*80}")
    print("FINAL COMPARISON RESULTS")
    print(f"{'='*80}\n")

    print(f"Model 1: {os.path.basename(args.prior1)}")
    # Determine metric name
    is_regression = hasattr(callback1, "score_history")
    metric_name = "R2 Score" if is_regression else "Accuracy"

    print(f"Model 1: {os.path.basename(args.prior1)}")
    print(f"  Final {metric_name}: {accuracy1:.4f}")
    print(f"  Final Loss: {callback1.loss_history[-1]:.4f}")
    print(f"  Training Time: {train_time1:.2f}s")
    print(f"  Inference Time: {inference_time1*1000:.2f}ms")
    print(f"  Parameters: {param_count1/1e6:.2f}M")
    print()
    print(f"Model 2: {os.path.basename(args.prior2)}")
    print(f"  Final {metric_name}: {accuracy2:.4f}")
    print(f"  Final Loss: {callback2.loss_history[-1]:.4f}")
    print(f"  Training Time: {train_time2:.2f}s")
    print(f"  Inference Time: {inference_time2*1000:.2f}ms")
    print(f"  Parameters: {param_count2/1e6:.2f}M")
    print()

    # Performance comparisons
    print("Performance Comparison:")
    accuracy_diff = accuracy1 - accuracy2
    better_accuracy = "Model 1" if accuracy_diff > 0 else "Model 2"
    print(
        f"  {metric_name} Difference: {abs(accuracy_diff):.4f} (Winner: {better_accuracy})"
    )

    train_speedup = train_time1 / train_time2
    faster_train = "Model 2" if train_speedup > 1 else "Model 1"
    print(f"  Training Speedup: {abs(train_speedup):.2f}x (Faster: {faster_train})")

    inference_speedup = inference_time1 / inference_time2
    faster_inference = "Model 2" if inference_speedup > 1 else "Model 1"
    print(
        f"  Inference Speedup: {abs(inference_speedup):.2f}x (Faster: {faster_inference})"
    )

    param_ratio = param_count1 / param_count2
    smaller_model = "Model 2" if param_ratio > 1 else "Model 1"
    print(f"  Parameter Ratio: {abs(param_ratio):.2f}x (Smaller: {smaller_model})")
    print(f"\n{'='*80}\n")

    # Create comparison plots
    prior1_name = os.path.basename(args.prior1).replace(".h5", "").replace("prior_", "")
    prior2_name = os.path.basename(args.prior2).replace(".h5", "").replace("prior_", "")
    plot_comparison(callback1, callback2, prior1_name, prior2_name, args.plot_output, metric_name=metric_name)

    return {
        "model1": {
            "prior": args.prior1,
            "accuracy": accuracy1,
            "model": model1,
            "loss_history": callback1.loss_history,
            "accuracy_history": callback1.score_history if is_regression else callback1.accuracy_history,
            "train_time": train_time1,
            "inference_time": inference_time1,
            "param_count": param_count1,
        },
        "model2": {
            "prior": args.prior2,
            "accuracy": accuracy2,
            "model": model2,
            "loss_history": callback2.loss_history,
            "accuracy_history": callback2.score_history if is_regression else callback2.accuracy_history,
            "train_time": train_time2,
            "inference_time": inference_time2,
            "param_count": param_count2,
        },
        "winner": better_accuracy,
        "performance": {
            "train_speedup": train_speedup,
            "inference_speedup": inference_speedup,
            "param_ratio": param_ratio,
        },
    }


if __name__ == "__main__":
    results = main()
