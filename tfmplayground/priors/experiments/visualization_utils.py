"""Visualization utilities for model comparison and evaluation."""

import os
from typing import List

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np
import requests
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.datasets import make_moons, make_circles
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from tfmplayground.interface import NanoTabPFNClassifier, NanoTabPFNRegressor
from tfmplayground.utils import get_default_device


def _resolve_plot_path(save_path: str) -> str:
    os.makedirs("plots", exist_ok=True)
    filename = os.path.basename(save_path)
    return os.path.join("plots", filename)


COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def plot_comparison_multi(
    callbacks,
    prior_names,
    save_path="comparison_plot.png",
    metric_name="Accuracy",
):
    """Create comparison plots for loss and metric curves for N runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss curves
    for cb, name in zip(callbacks, prior_names):
        ax1.plot(cb.epoch_history, cb.loss_history, label=name, linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training Loss Comparison", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Plot metric curves (skip None values)
    for cb, name in zip(callbacks, prior_names):
        metric_history = (
            cb.score_history if hasattr(cb, "score_history") else cb.accuracy_history
        )
        xs, ys = [], []
        for e, m in zip(cb.epoch_history, metric_history):
            if m is None:
                continue
            xs.append(e)
            ys.append(m)
        ax2.plot(xs, ys, label=name, linewidth=2)

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel(metric_name, fontsize=12)
    ax2.set_title(
        f"Validation {metric_name} Comparison", fontsize=14, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Legends outside
    ax1.legend(fontsize=9, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
    ax2.legend(fontsize=9, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)

    save_path = _resolve_plot_path(save_path)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Comparison plot saved to: {save_path}")

    try:
        plt.show()
    except:
        pass

    plt.close()


def plot_per_task_comparison(
    run_records: List,
    output_path: str = "comparison_per_task.png",
    metric_name: str = "Accuracy",
    k_tasks: int = 0,
):
    """Create grouped bar chart comparing per-task scores across models."""
    if not run_records:
        return

    all_tasks = set()
    final_task_scores = []
    for record in run_records:
        task_scores = record.get("per_task_scores", {})
        if not task_scores:
            continue
        final_scores = {}
        for task, vals in task_scores.items():
            if isinstance(vals, list):
                final_val = None
                for v in reversed(vals):
                    if v is not None:
                        final_val = float(v)
                        break
                final_scores[task] = 0.0 if final_val is None else final_val
            else:
                final_scores[task] = 0.0 if vals is None else float(vals)
        final_task_scores.append((record, final_scores))
        all_tasks.update(final_scores.keys())

    if not all_tasks:
        print("⚠️  No per-task scores found to plot")
        return

    all_tasks = sorted(all_tasks)
    if k_tasks and k_tasks > 0 and len(all_tasks) > k_tasks:
        ranges = []
        for task_name in all_tasks:
            vals = []
            for _, scores in final_task_scores:
                v = scores.get(task_name, None)
                if v is not None:
                    vals.append(float(v))
            if vals:
                ranges.append((max(vals) - min(vals), task_name))
        ranges.sort(reverse=True)
        all_tasks = [task_name for _, task_name in ranges[:k_tasks]]

    n_tasks = len(all_tasks)
    n_models = len(final_task_scores)

    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 2), 6))
    x = range(n_tasks)
    bar_width = 0.8 / max(1, n_models)

    for i, (record, scores_dict) in enumerate(final_task_scores):
        scores = [scores_dict.get(task, 0.0) for task in all_tasks]
        offset = (i - n_models / 2 + 0.5) * bar_width
        color = COLORS[i % len(COLORS)]
        short_name = record.get("prior_name", record.get("model_name", "Model"))

        bars = ax.bar(
            [xi + offset for xi in x],
            scores,
            bar_width,
            label=short_name,
            color=color,
            alpha=0.8,
        )

        for bar, score in zip(bars, scores):
            if score != 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{score:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"Per-Task {metric_name} Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(all_tasks, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=9, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
    ax.grid(True, alpha=0.3, axis="y")

    output_path = _resolve_plot_path(output_path)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📊 Per-task comparison plot saved to: {output_path}")

    plt.close()


def plot_time_budget_metrics(
    run_records,
    metric_name="Accuracy",
    output_prefix="comparison_plot",
    n_budgets=20,
):
    """Plot mean metric, win share, and rank as a function of time budget."""
    if not run_records:
        return

    model_series = []
    epoch_times = []
    for record in run_records:
        metric_history = record.get("metric_history", [])
        loss_history = record.get("loss_history", [])
        train_time = record.get("train_time", 0.0)
        if not loss_history:
            continue

        epochs = len(loss_history)
        epoch_time = train_time / max(epochs, 1)
        epoch_times.append(epoch_time)

        filled = []
        last_val = None
        for val in metric_history:
            if val is None:
                filled.append(last_val)
            else:
                last_val = float(val)
                filled.append(last_val)

        model_series.append(
            {
                "name": record.get("prior_name", record.get("model_name", "Model")),
                "epoch_time": epoch_time,
                "metrics": filled,
            }
        )

    if not model_series:
        return

    min_epoch_time = min(epoch_times) if epoch_times else 0.01
    max_time = max(r.get("train_time", 0.0) for r in run_records)
    min_time = max(min_epoch_time, 1e-3)
    if max_time <= min_time:
        budgets = np.linspace(min_time, max_time + 1e-3, n_budgets)
    else:
        budgets = np.logspace(np.log10(min_time), np.log10(max_time), n_budgets)

    metric_by_budget = {m["name"]: [] for m in model_series}
    win_share_by_budget = {m["name"]: [] for m in model_series}
    rank_by_budget = {m["name"]: [] for m in model_series}

    for budget in budgets:
        current_metrics = []
        for model in model_series:
            epoch_idx = int(budget / max(model["epoch_time"], 1e-9)) - 1
            epoch_idx = max(0, min(epoch_idx, len(model["metrics"]) - 1))
            metric_val = model["metrics"][epoch_idx]
            current_metrics.append(metric_val)
            metric_by_budget[model["name"]].append(metric_val)

        valid = [m for m in current_metrics if m is not None]
        if not valid:
            for model in model_series:
                win_share_by_budget[model["name"]].append(np.nan)
                rank_by_budget[model["name"]].append(np.nan)
            continue

        max_val = max(valid)
        winners = [i for i, m in enumerate(current_metrics) if m == max_val]
        win_share = 1.0 / len(winners) if winners else 0.0
        for i, model in enumerate(model_series):
            win_share_by_budget[model["name"]].append(
                win_share if i in winners else 0.0
            )

        sorted_vals = sorted(
            [(i, m) for i, m in enumerate(current_metrics) if m is not None],
            key=lambda x: x[1],
            reverse=True,
        )
        ranks = [np.nan] * len(model_series)
        rank = 1
        i = 0
        while i < len(sorted_vals):
            j = i
            while j < len(sorted_vals) and sorted_vals[j][1] == sorted_vals[i][1]:
                j += 1
            avg_rank = (rank + (rank + (j - i) - 1)) / 2
            for k in range(i, j):
                ranks[sorted_vals[k][0]] = avg_rank
            rank += j - i
            i = j

        for i, model in enumerate(model_series):
            rank_by_budget[model["name"]].append(ranks[i])

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, model in enumerate(model_series):
        name = model["name"]
        ax.plot(
            budgets, metric_by_budget[name], label=name, color=COLORS[i % len(COLORS)]
        )
    ax.set_xscale("log")
    ax.set_xlabel("Time Budget (s)", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"Mean {metric_name} / Time Budget", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
    output_path = _resolve_plot_path(f"{output_prefix}_time_budget_metric.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📊 Mean {metric_name} time budget plot saved to: {output_path}")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, model in enumerate(model_series):
        name = model["name"]
        ax.plot(
            budgets,
            win_share_by_budget[name],
            label=name,
            color=COLORS[i % len(COLORS)],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Time Budget (s)", fontsize=12)
    ax.set_ylabel("Win Share", fontsize=12)
    ax.set_title(
        f"Mean {metric_name} Wins / Time Budget", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
    output_path = _resolve_plot_path(f"{output_prefix}_time_budget_wins.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📊 Mean {metric_name} wins time budget plot saved to: {output_path}")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, model in enumerate(model_series):
        name = model["name"]
        ax.plot(
            budgets, rank_by_budget[name], label=name, color=COLORS[i % len(COLORS)]
        )
    ax.set_xscale("log")
    ax.set_xlabel("Time Budget (s)", fontsize=12)
    ax.set_ylabel("Rank (1 = best)", fontsize=12)
    ax.set_title(
        f"Mean {metric_name} Rank / Time Budget", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
    output_path = _resolve_plot_path(f"{output_prefix}_time_budget_rank.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📊 Mean {metric_name} rank time budget plot saved to: {output_path}")
    plt.close()


def generate_toy_dataset(name, n_samples=200, noise=0.2, random_state=42):
    """Generate scikit-learn toy datasets for decision boundary visualization."""
    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif name == "circles":
        X, y = make_circles(
            n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def plot_decision_boundary(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    ax,
    title="Model",
    resolution=100,
    device=None,
):
    """Plot decision boundary with probability zones."""
    if device is None:
        device = get_default_device()

    classifier = NanoTabPFNClassifier(model, device)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    # Create mesh grid for decision boundary
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z_proba = classifier.predict_proba(grid_points)

    n_classes = len(np.unique(y_train))
    colors = np.array([[0.8, 0.2, 0.2], [0.2, 0.2, 0.8]])[:n_classes]

    # Blend colors based on probabilities
    rgb_image = np.zeros((resolution, resolution, 3))
    for c in range(n_classes):
        proba_c = Z_proba[:, c].reshape(xx.shape)
        for channel in range(3):
            rgb_image[:, :, channel] += proba_c * colors[c, channel]

    rgb_image = np.clip(rgb_image, 0, 1)

    ax.imshow(
        rgb_image,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        aspect="auto",
        alpha=0.7,
    )

    # Add contour lines
    for c in range(n_classes):
        proba_c = Z_proba[:, c].reshape(xx.shape)
        ax.contour(
            xx,
            yy,
            proba_c,
            levels=[0.5],
            colors=[colors[c]],
            linewidths=2,
            linestyles="solid",
        )

    cmap_bold = ListedColormap(["#CC0000", "#0000CC"][:n_classes])

    # Plot training points
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap=cmap_bold,
        edgecolors="white",
        s=30,
        linewidth=1,
        marker="o",
        alpha=0.8,
    )

    # Plot test points
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap=cmap_bold,
        edgecolors="black",
        s=25,
        linewidth=1,
        marker="s",
        alpha=0.6,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x1", fontsize=9)
    ax.set_ylabel("x2", fontsize=9)
    ax.set_title(f"{title}\nAcc: {accuracy:.1%}", fontsize=10, fontweight="bold")

    return accuracy


def plot_sklearn_decision_boundary(
    sklearn_model, X_train, X_test, y_train, y_test, ax, title="Model", resolution=100
):
    """Plot decision boundary for sklearn models."""
    sklearn_model.fit(X_train, y_train)
    y_pred = sklearn_model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    if hasattr(sklearn_model, "predict_proba"):
        Z_proba = sklearn_model.predict_proba(grid_points)
    else:
        Z_pred = sklearn_model.predict(grid_points)
        Z_proba = np.zeros((len(Z_pred), 2))
        Z_proba[np.arange(len(Z_pred)), Z_pred] = 1.0

    n_classes = 2
    colors = np.array([[0.8, 0.2, 0.2], [0.2, 0.2, 0.8]])

    rgb_image = np.zeros((resolution, resolution, 3))
    for c in range(n_classes):
        proba_c = Z_proba[:, c].reshape(xx.shape)
        for channel in range(3):
            rgb_image[:, :, channel] += proba_c * colors[c, channel]

    rgb_image = np.clip(rgb_image, 0, 1)

    ax.imshow(
        rgb_image,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        aspect="auto",
        alpha=0.7,
    )

    for c in range(n_classes):
        proba_c = Z_proba[:, c].reshape(xx.shape)
        ax.contour(
            xx,
            yy,
            proba_c,
            levels=[0.5],
            colors=[colors[c]],
            linewidths=2,
            linestyles="solid",
        )

    cmap_bold = ListedColormap(["#CC0000", "#0000CC"])

    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap=cmap_bold,
        edgecolors="white",
        s=30,
        linewidth=1,
        marker="o",
        alpha=0.8,
    )
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap=cmap_bold,
        edgecolors="black",
        s=25,
        linewidth=1,
        marker="s",
        alpha=0.6,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x1", fontsize=9)
    ax.set_ylabel("x2", fontsize=9)
    ax.set_title(f"{title}\nAcc: {accuracy:.1%}", fontsize=10, fontweight="bold")

    return accuracy


def generate_toy_regression_dataset(name, n_samples=100, noise=0.1, random_state=42):
    """Generate simple 1D regression datasets."""
    np.random.seed(random_state)

    # Generate X values
    X = np.linspace(-3, 3, n_samples)

    if name == "sinusoidal":
        y = np.sin(X * 2)
    elif name == "linear":
        y = 0.5 * X + 1.0
    elif name == "step":
        y = np.where(X < -1, -1, np.where(X < 1, 0, 1)).astype(float)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Add noise
    y = y + np.random.normal(0, noise, n_samples)

    # Reshape for sklearn compatibility
    X = X.reshape(-1, 1)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def plot_regression_prediction(
    model, X_train, X_test, y_train, y_test, ax, title="Model", device=None
):
    """Plot regression predictions with prediction curve."""
    if device is None:
        device = get_default_device()

    # Load bucket edges if needed
    buckets_path = "checkpoints/nanotabpfn_regressor_buckets.pth"
    if not os.path.isfile(buckets_path):
        print(f"Downloading bucket edges to {buckets_path}...")
        os.makedirs(os.path.dirname(buckets_path), exist_ok=True)
        response = requests.get(
            "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor_buckets.pth"
        )
        with open(buckets_path, "wb") as f:
            f.write(response.content)

    bucket_edges = torch.load(buckets_path, map_location=device)
    dist = FullSupportBarDistribution(bucket_edges).float().to(device)

    regressor = NanoTabPFNRegressor(model, dist=dist, device=device)
    regressor.fit(X_train, y_train)

    # Predictions on test set
    y_pred_test = regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)

    # Create smooth curve for visualization
    X_plot = np.linspace(X_train.min(), X_train.max(), 200).reshape(-1, 1)
    y_plot = regressor.predict(X_plot)

    # Plot the prediction curve
    ax.plot(X_plot, y_plot, "b-", linewidth=2, label="Prediction", alpha=0.8)

    # Plot training points
    ax.scatter(
        X_train,
        y_train,
        c="green",
        s=30,
        alpha=0.6,
        edgecolors="white",
        linewidth=1,
        label="Train",
        marker="o",
    )

    # Plot test points
    ax.scatter(
        X_test,
        y_test,
        c="red",
        s=25,
        alpha=0.6,
        edgecolors="black",
        linewidth=1,
        label="Test",
        marker="s",
    )

    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.set_title(f"{title}\nR²: {r2:.3f}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    return r2


def plot_sklearn_regression_prediction(
    sklearn_model, X_train, X_test, y_train, y_test, ax, title="Model"
):
    """Plot regression predictions for sklearn models."""
    sklearn_model.fit(X_train, y_train)

    # Predictions on test set
    y_pred_test = sklearn_model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)

    # Create smooth curve for visualization
    X_plot = np.linspace(X_train.min(), X_train.max(), 200).reshape(-1, 1)
    y_plot = sklearn_model.predict(X_plot)

    # Plot the prediction curve
    ax.plot(X_plot, y_plot, "b-", linewidth=2, label="Prediction", alpha=0.8)

    # Plot training points
    ax.scatter(
        X_train,
        y_train,
        c="green",
        s=30,
        alpha=0.6,
        edgecolors="white",
        linewidth=1,
        label="Train",
        marker="o",
    )

    # Plot test points
    ax.scatter(
        X_test,
        y_test,
        c="red",
        s=25,
        alpha=0.6,
        edgecolors="black",
        linewidth=1,
        label="Test",
        marker="s",
    )

    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.set_title(f"{title}\nR²: {r2:.3f}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    return r2


def plot_all_regression_predictions(
    run_records,
    datasets=["sinusoidal", "linear", "step"],
    n_samples=100,
    noise=0.1,
    seed=42,
    output_path="regression_predictions_comparison.png",
):
    """Create grid showing regression predictions for all models on toy datasets."""
    if not run_records:
        return

    device = get_default_device()

    # Add baseline models
    baseline_models = [
        {"name": "Linear Regression", "model": LinearRegression()},
        {"name": "SVR (RBF)", "model": SVR(kernel="rbf")},
    ]

    n_tabpfn_models = len(run_records)
    n_total_models = n_tabpfn_models + len(baseline_models)
    n_datasets = len(datasets)

    fig, axes = plt.subplots(
        n_datasets, n_total_models, figsize=(4 * n_total_models, 4 * n_datasets)
    )

    if n_datasets == 1 and n_total_models == 1:
        axes = np.array([[axes]])
    elif n_datasets == 1:
        axes = axes.reshape(1, -1)
    elif n_total_models == 1:
        axes = axes.reshape(-1, 1)

    for dataset_idx, dataset_name in enumerate(datasets):
        X_train, X_test, y_train, y_test = generate_toy_regression_dataset(
            dataset_name, n_samples=n_samples, noise=noise, random_state=seed
        )

        # Plot baseline models first
        for baseline_idx, baseline_info in enumerate(baseline_models):
            if dataset_idx == 0:
                title = baseline_info["name"]
            else:
                title = ""

            ax = axes[dataset_idx, baseline_idx]
            plot_sklearn_regression_prediction(
                baseline_info["model"],
                X_train,
                X_test,
                y_train,
                y_test,
                ax,
                title=title,
            )

            if dataset_idx == 0 and baseline_idx == 0:
                ax.text(
                    -0.3,
                    0.5,
                    dataset_name.capitalize(),
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    rotation=90,
                )

        # Plot TabPFN models
        for model_idx, record in enumerate(run_records):
            model = record["model"]
            prior_name = record["prior_name"]

            if dataset_idx == 0:
                title = prior_name
            else:
                title = ""

            ax = axes[dataset_idx, model_idx + len(baseline_models)]
            plot_regression_prediction(
                model, X_train, X_test, y_train, y_test, ax, title=title, device=device
            )

            if model_idx == 0 and len(baseline_models) == 0:
                ax.text(
                    -0.3,
                    0.5,
                    dataset_name.capitalize(),
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    rotation=90,
                )

    plt.suptitle(
        "Regression Predictions on Toy Datasets",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    output_path = _resolve_plot_path(output_path)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Regression predictions plot saved to: {output_path}")

    try:
        plt.show()
    except:
        pass

    plt.close()


def plot_all_decision_boundaries(
    run_records,
    datasets=["moons", "circles"],
    n_samples=200,
    noise=0.2,
    seed=42,
    output_path="decision_boundaries_comparison.png",
):
    """Create grid showing decision boundaries for all models on all toy datasets."""
    if not run_records:
        return

    device = get_default_device()

    # Add baseline models
    baseline_models = [
        {"name": "Logistic Regression", "model": LogisticRegression(max_iter=1000)},
        {"name": "SVM (RBF)", "model": SVC(kernel="rbf", probability=True)},
    ]

    n_tabpfn_models = len(run_records)
    n_total_models = n_tabpfn_models + len(baseline_models)
    n_datasets = len(datasets)

    fig, axes = plt.subplots(
        n_datasets, n_total_models, figsize=(4 * n_total_models, 4 * n_datasets)
    )

    if n_datasets == 1 and n_total_models == 1:
        axes = np.array([[axes]])
    elif n_datasets == 1:
        axes = axes.reshape(1, -1)
    elif n_total_models == 1:
        axes = axes.reshape(-1, 1)

    for dataset_idx, dataset_name in enumerate(datasets):
        X_train, X_test, y_train, y_test = generate_toy_dataset(
            dataset_name, n_samples=n_samples, noise=noise, random_state=seed
        )

        # Plot baseline models first
        for baseline_idx, baseline_info in enumerate(baseline_models):
            if dataset_idx == 0:
                title = baseline_info["name"]
            else:
                title = ""

            ax = axes[dataset_idx, baseline_idx]
            plot_sklearn_decision_boundary(
                baseline_info["model"],
                X_train,
                X_test,
                y_train,
                y_test,
                ax,
                title=title,
            )

            if dataset_idx == 0 and baseline_idx == 0:
                ax.text(
                    -0.3,
                    0.5,
                    dataset_name.capitalize(),
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    rotation=90,
                )

        # Plot TabPFN models
        for model_idx, record in enumerate(run_records):
            model = record["model"]
            prior_name = record["prior_name"]

            if dataset_idx == 0:
                title = prior_name
            else:
                title = ""

            ax = axes[dataset_idx, model_idx + len(baseline_models)]
            plot_decision_boundary(
                model, X_train, X_test, y_train, y_test, ax, title=title, device=device
            )

            # Add dataset label on the left (only for first baseline model now)
            if model_idx == 0 and len(baseline_models) == 0:
                ax.text(
                    -0.3,
                    0.5,
                    dataset_name.capitalize(),
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    rotation=90,
                )

    plt.suptitle(
        "Decision Boundaries on Toy Datasets", fontsize=14, fontweight="bold", y=0.995
    )
    output_path = _resolve_plot_path(output_path)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Decision boundaries plot saved to: {output_path}")

    try:
        plt.show()
    except:
        pass

    plt.close()
