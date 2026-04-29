"""Visualization utilities for model comparison and evaluation."""

import os
from typing import List

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, root_mean_squared_error
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
    "#60c8d3",  # cyan
    "#29043E",  # dark purple
    "#063042",  # ocean
    "#274720",  # forest green
]


def _get_n_colors(n: int) -> List[str]:
    """Return colors by cycling the configured palette."""
    return [COLORS[i % len(COLORS)] for i in range(n)]


def plot_comparison_multi(
    callbacks,
    prior_names,
    save_path="comparison_plot.png",
    metric_name="Accuracy",
):
    """Create comparison plots for loss and metric curves for N runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    colors = _get_n_colors(len(prior_names))

    # Plot loss curves
    for i, (cb, name) in enumerate(zip(callbacks, prior_names)):
        ax1.plot(
            cb.epoch_history,
            cb.loss_history,
            label=name,
            linewidth=2,
            color=colors[i],
        )
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training Loss Comparison", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Plot metric curves (skip None values)
    for i, (cb, name) in enumerate(zip(callbacks, prior_names)):
        metric_history = (
            cb.roc_auc_history if hasattr(cb, "roc_auc_history") else cb.rmse_history
        )
        xs, ys = [], []
        for e, m in zip(cb.epoch_history, metric_history):
            if m is None:
                continue
            xs.append(e)
            ys.append(m)
        ax2.plot(xs, ys, label=name, linewidth=2, color=colors[i])

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
        task_scores = record.get("final_task_scores", {})
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
    colors = _get_n_colors(n_models)

    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 2), 6))
    x = range(n_tasks)
    bar_width = 0.8 / max(1, n_models)

    for i, (record, scores_dict) in enumerate(final_task_scores):
        scores = [scores_dict.get(task, 0.0) for task in all_tasks]
        offset = (i - n_models / 2 + 0.5) * bar_width
        color = colors[i]
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
    for run_idx, record in enumerate(run_records):
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
                "key": f"run_{run_idx}",
                "name": record.get("prior_name", record.get("model_name", "Model")),
                "epoch_time": epoch_time,
                "metrics": filled,
            }
        )

    if not model_series:
        return

    seen = {}
    for model in model_series:
        name = model["name"]
        seen[name] = seen.get(name, 0) + 1
        model["display_name"] = name if seen[name] == 1 else f"{name} ({seen[name]})"

    colors = _get_n_colors(len(model_series))

    min_epoch_time = min(epoch_times) if epoch_times else 0.01
    max_time = max(r.get("train_time", 0.0) for r in run_records)
    min_time = max(min_epoch_time, 1e-3)
    if max_time <= min_time:
        budgets = np.linspace(min_time, max_time + 1e-3, n_budgets)
    else:
        budgets = np.logspace(np.log10(min_time), np.log10(max_time), n_budgets)

    metric_by_budget = {m["key"]: [] for m in model_series}
    win_share_by_budget = {m["key"]: [] for m in model_series}
    rank_by_budget = {m["key"]: [] for m in model_series}

    for budget in budgets:
        current_metrics = []
        for model in model_series:
            epoch_idx = int(budget / max(model["epoch_time"], 1e-9)) - 1
            epoch_idx = max(0, min(epoch_idx, len(model["metrics"]) - 1))
            metric_val = model["metrics"][epoch_idx]
            current_metrics.append(metric_val)
            metric_by_budget[model["key"]].append(metric_val)

        valid = [m for m in current_metrics if m is not None]
        if not valid:
            for model in model_series:
                win_share_by_budget[model["key"]].append(np.nan)
                rank_by_budget[model["key"]].append(np.nan)
            continue

        max_val = max(valid)
        winners = [i for i, m in enumerate(current_metrics) if m == max_val]
        win_share = 1.0 / len(winners) if winners else 0.0
        for i, model in enumerate(model_series):
            win_share_by_budget[model["key"]].append(
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
            rank_by_budget[model["key"]].append(ranks[i])

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, model in enumerate(model_series):
        key = model["key"]
        ax.plot(
            budgets, metric_by_budget[key], label=model["display_name"], color=colors[i]
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
        key = model["key"]
        ax.plot(
            budgets,
            win_share_by_budget[key],
            label=model["display_name"],
            color=colors[i],
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
        key = model["key"]
        ax.plot(
            budgets, rank_by_budget[key], label=model["display_name"], color=colors[i]
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
    roc_auc = roc_auc_score(y_test, y_pred)

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
    ax.set_title(f"{title}\nROC-AUC: {roc_auc:.1%}", fontsize=10, fontweight="bold")

    return roc_auc


def plot_sklearn_decision_boundary(
    sklearn_model, X_train, X_test, y_train, y_test, ax, title="Model", resolution=100
):
    """Plot decision boundary for sklearn models."""
    sklearn_model.fit(X_train, y_train)
    y_pred = sklearn_model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_pred)

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
    ax.set_title(f"{title}\nROC-AUC: {roc_auc:.1%}", fontsize=10, fontweight="bold")

    return roc_auc


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
    model, X_train, X_test, y_train, y_test, ax, title="Model", device=None, dist=None
):
    """Plot regression predictions with prediction curve."""
    if device is None:
        device = get_default_device()

    if dist is None:
        raise ValueError("A distribution (dist) must be provided for the regressor.")

    regressor = NanoTabPFNRegressor(model, dist=dist, device=device)
    regressor.fit(X_train, y_train)

    # Predictions on test set
    y_pred_test = regressor.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred_test)

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
    ax.set_title(f"{title}\nRMSE: {rmse:.3f}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    return rmse


def plot_sklearn_regression_prediction(
    sklearn_model, X_train, X_test, y_train, y_test, ax, title="Model"
):
    """Plot regression predictions for sklearn models."""
    sklearn_model.fit(X_train, y_train)

    # Predictions on test set
    y_pred_test = sklearn_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred_test)

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
    ax.set_title(f"{title}\nRMSE: {rmse:.3f}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    return rmse


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

    n_trained_models = len(run_records)
    n_total_models = n_trained_models + len(baseline_models)
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

        # Plot trained models
        for model_idx, record in enumerate(run_records):
            model = record["model"]
            dist = record.get("dist")
            prior_name = record["prior_name"]

            if dataset_idx == 0:
                title = prior_name
            else:
                title = ""

            ax = axes[dataset_idx, model_idx + len(baseline_models)]
            plot_regression_prediction(
                model, X_train, X_test, y_train, y_test, ax, title=title, device=device, dist=dist
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

    n_trained_models = len(run_records)
    n_total_models = n_trained_models + len(baseline_models)
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

        # Plot trained models
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





def _normalize_results_variable_folds(metrics_payload):
    models = metrics_payload["models"]
    model_count = len(models)
    datasets = list(models[0]["task_scores"].keys())
    n_epochs = len(models[0]["task_scores"][datasets[0]])

    normalized_scores = {m: [] for m in range(model_count)}

    for e in range(n_epochs):

        per_model_dataset_means = {m: [] for m in range(model_count)}

        for d in datasets:

            fold_count = len(models[0]["task_scores"][d][e])

            per_model_fold_scores = {m: [] for m in range(model_count)}

            for f in range(fold_count):

                raw = np.array([
                    models[m]["task_scores"][d][e][f]
                    for m in range(model_count)
                ])

                vmin = raw.min()
                vmax = raw.max()

                if vmax == vmin:
                    norm = np.ones(model_count) * 0.5
                else:
                    # RMSE → lower is better
                    norm = (vmax - raw) / (vmax - vmin)

                for m in range(model_count):
                    per_model_fold_scores[m].append(norm[m])

            # average folds → dataset-level score
            for m in range(model_count):
                per_model_dataset_means[m].append(
                    np.mean(per_model_fold_scores[m])
                )

        # average datasets → final epoch score
        for m in range(model_count):
            normalized_scores[m].append(
                np.mean(per_model_dataset_means[m])
            )

    return normalized_scores


def plot_per_fold_normalized_averaged_metrics(
    metrics_payload,
    metric_name="Accuracy",
    output_path="per_fold_normalized_comparison.png",
):
    """Plot per-fold normalized averaged metrics for all models."""

    if not metrics_payload["models"]:
        return

    # Compute normalized scores (per model, per epoch)
    normalized_scores = _normalize_results_variable_folds(metrics_payload)

    models = metrics_payload["models"]

    plt.figure(figsize=(8, 5))

    for model in models:
        m_idx = model["model_index"] - 1  # adjust if model_index starts from 1
        model_name = model["model_name"]
        epochs = model["epochs"]

        if m_idx not in normalized_scores:
            continue

        scores = normalized_scores[m_idx]

        # Safety check in case lengths differ
        min_len = min(len(epochs), len(scores))
        epochs = epochs[:min_len]
        scores = scores[:min_len]

        plt.plot(
            epochs,
            scores,
            marker="o",
            label=model_name,
        )

    plt.xlabel("Epoch")
    plt.ylabel(f"Normalized {metric_name}")
    plt.title(f"Per-Fold Normalized Averaged {metric_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path = _resolve_plot_path(output_path)
    plt.savefig(output_path)
    plt.close()


def plot_tabarena_performance_heatmap(
    perf_matrix: np.ndarray,
    prior_names: list[str],
    dataset_names: list[str],
    metric_name: str = "ROC-AUC",
    output_path: str = "tabarena_performance_heatmap.png",
):
    """Plot a heatmap of prior (rows) vs TabArena dataset (columns) performance.

    Args:
        perf_matrix: 2-D array of shape (n_priors, n_datasets).
        prior_names: Names for each row.
        dataset_names: Names for each column.
        metric_name: Label shown on the colour-bar.
        output_path: File path for the saved figure.
    """
    n_priors, n_datasets = perf_matrix.shape

    fig_width = max(10, n_datasets * 0.7 + 3)
    fig_height = max(4, n_priors * 0.6 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(perf_matrix, aspect="auto", cmap="RdYlGn")

    ax.set_xticks(range(n_datasets))
    ax.set_xticklabels(dataset_names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n_priors))
    ax.set_yticklabels(prior_names, fontsize=10)

    # Annotate cells
    for i in range(n_priors):
        for j in range(n_datasets):
            val = perf_matrix[i, j]
            if np.isnan(val):
                text = "—"
            else:
                text = f"{val:.3f}"
            # Pick contrasting text colour
            norm_val = (val - np.nanmin(perf_matrix)) / (
                np.nanmax(perf_matrix) - np.nanmin(perf_matrix) + 1e-9
            ) if not np.isnan(val) else 0.5
            text_color = "white" if norm_val < 0.35 or norm_val > 0.85 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label(metric_name, fontsize=11)

    ax.set_title(
        f"Prior vs TabArena — {metric_name}",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("TabArena Dataset", fontsize=12)
    ax.set_ylabel("Prior", fontsize=12)

    output_path = _resolve_plot_path(output_path)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📊 TabArena performance heatmap saved to: {output_path}")
    plt.close()


def plot_tabarena_normalized_heatmap(
    perf_matrix: np.ndarray,
    prior_names: list[str],
    dataset_names: list[str],
    metric_name: str = "ROC-AUC",
    output_path: str = "tabarena_normalized_heatmap.png",
):
    """Plot a min/max-normalized heatmap of prior vs TabArena dataset performance.

    For each dataset (column), scores are normalized so that the worst prior
    maps to 0 and the best prior maps to 1.  This removes dataset difficulty
    as a confound and highlights relative differences between priors.

    Args:
        perf_matrix: 2-D array of shape (n_priors, n_datasets).
        prior_names: Names for each row.
        dataset_names: Names for each column.
        metric_name: Original metric label (used in the title).
        output_path: File path for the saved figure.
    """
    n_priors, n_datasets = perf_matrix.shape

    # Min/max normalize per dataset (column-wise)
    col_min = np.nanmin(perf_matrix, axis=0, keepdims=True)
    col_max = np.nanmax(perf_matrix, axis=0, keepdims=True)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0  # avoid division by zero for constant columns
    norm_matrix = (perf_matrix - col_min) / col_range

    # Sort rows by mean normalized score (best prior on top)
    row_means = np.nanmean(norm_matrix, axis=1)
    sorted_idx = np.argsort(row_means)[::-1]
    norm_matrix = norm_matrix[sorted_idx]
    prior_names = [prior_names[i] for i in sorted_idx]

    fig_width = max(10, n_datasets * 0.7 + 3)
    fig_height = max(4, n_priors * 0.6 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(norm_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(n_datasets))
    ax.set_xticklabels(dataset_names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n_priors))
    ax.set_yticklabels(prior_names, fontsize=10)

    # Annotate cells
    for i in range(n_priors):
        for j in range(n_datasets):
            val = norm_matrix[i, j]
            if np.isnan(val):
                text = "—"
            else:
                text = f"{val:.2f}"
            text_color = "white" if (not np.isnan(val) and (val < 0.35 or val > 0.85)) else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label("Normalized Score (0 = worst, 1 = best)", fontsize=11)

    ax.set_title(
        f"Prior vs TabArena — Normalized {metric_name} (per-dataset min/max)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("TabArena Dataset", fontsize=12)
    ax.set_ylabel("Prior", fontsize=12)

    output_path = _resolve_plot_path(output_path)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📊 Normalized TabArena heatmap saved to: {output_path}")
    plt.close()

def plot_prior_correlation_heatmap(
    perf_matrix: np.ndarray,
    prior_names: list[str],
    output_path: str = "prior_correlation_heatmap.png",
):
    """Plot a prior-vs-prior Pearson correlation heatmap.

    Correlation is computed between rows of *perf_matrix* (each row is
    a prior's performance vector across datasets).

    Args:
        perf_matrix: 2-D array of shape (n_priors, n_datasets).
        prior_names: Names for each prior.
        output_path: File path for the saved figure.
    """
    n_priors = perf_matrix.shape[0]

    # Mask NaN columns so corrcoef doesn't produce all-NaN results
    valid_cols = ~np.any(np.isnan(perf_matrix), axis=0)
    clean_matrix = perf_matrix[:, valid_cols]

    if clean_matrix.shape[1] < 2:
        print("⚠️  Not enough valid datasets to compute prior correlation.")
        return

    corr = np.corrcoef(clean_matrix)
    if n_priors > 1:
        off_diag = corr[~np.eye(n_priors, dtype=bool)]
        off_diag = off_diag[np.isfinite(off_diag)]
    else:
        off_diag = np.array([], dtype=float)

    if off_diag.size:
        # use a wider positive band so the plot keeps contrast without becoming uniformly red.
        clip_vmin = 0.65
        clip_vmax = 1.0
    else:
        clip_vmin, clip_vmax = -1.0, 1.0

    if (not np.isfinite(clip_vmin)) or (not np.isfinite(clip_vmax)):
        clip_vmin, clip_vmax = -1.0, 1.0

    if clip_vmax - clip_vmin < 1e-3:
        center = float(np.mean(off_diag)) if off_diag.size else 0.0
        half = 0.05
        clip_vmin = max(-1.0, center - half)
        clip_vmax = min(1.0, center + half)

    corr_plot = np.clip(corr, clip_vmin, clip_vmax)

    fig_size = max(5, n_priors * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(corr_plot, cmap="coolwarm", vmin=clip_vmin, vmax=clip_vmax, aspect="equal")

    ax.set_xticks(range(n_priors))
    ax.set_xticklabels(prior_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n_priors))
    ax.set_yticklabels(prior_names, fontsize=10)

    # Annotate cells
    for i in range(n_priors):
        for j in range(n_priors):
            val = corr[i, j]
            text_color = "white"
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=9, fontweight="bold", color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label(
        f"Pearson Correlation (clipped to [{clip_vmin:.2f}, {clip_vmax:.2f}])",
        fontsize=11,
    )

    ax.set_title(
        "Prior vs Prior — Performance Correlation",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    output_path = _resolve_plot_path(output_path)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📊 Prior correlation heatmap saved to: {output_path}")
    plt.close()


def compute_performance_similarity_matrix(perf_matrix: np.ndarray) -> np.ndarray | None:
    """Compute prior-prior performance similarity from TabArena matrix."""
    valid_cols = ~np.any(np.isnan(perf_matrix), axis=0)
    clean_matrix = perf_matrix[:, valid_cols]

    if clean_matrix.shape[1] < 2:
        return None

    corr = np.corrcoef(clean_matrix)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def plot_data_similarity_heatmap(
    data_similarity_matrix: np.ndarray,
    prior_names: list[str],
    output_path: str = "prior_data_similarity_heatmap.png",
):
    """Plot a prior-vs-prior heatmap for data similarity."""
    n_priors = data_similarity_matrix.shape[0]
    if data_similarity_matrix.shape[0] != data_similarity_matrix.shape[1]:
        raise ValueError("data_similarity_matrix must be square")
    if len(prior_names) != n_priors:
        raise ValueError("prior_names length must match matrix shape")

    fig_size = max(5, n_priors * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(data_similarity_matrix, cmap="coolwarm", vmin=0.0, vmax=1.0, aspect="equal")

    ax.set_xticks(range(n_priors))
    ax.set_xticklabels(prior_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n_priors))
    ax.set_yticklabels(prior_names, fontsize=10)
    ax.set_xlabel("Prior", fontsize=12)
    ax.set_ylabel("Prior", fontsize=12)
    ax.set_title(
        "Prior vs Prior - Data Similarity",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    for i in range(n_priors):
        for j in range(n_priors):
            val = data_similarity_matrix[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Distance-based similarity in standardized meta-feature space", fontsize=11)

    output_path = _resolve_plot_path(output_path)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📊 Prior data similarity heatmap saved to: {output_path}")
    plt.close()


def compute_data_vs_performance_stats(
    data_similarity_matrix: np.ndarray,
    performance_similarity_matrix: np.ndarray,
    prior_names: list[str],
    top_k: int = 10,
    high_similarity_threshold: float = 0.6,
) -> dict:
    """Compute pairwise data/performance agreement statistics.

    Returns a dict with ranked_pairs, top mismatch categories, and
    thresholds that the plotting function can consume directly.
    """
    if data_similarity_matrix.shape != performance_similarity_matrix.shape:
        raise ValueError("data and performance similarity matrices must have the same shape")

    n_priors = data_similarity_matrix.shape[0]
    if n_priors < 2:
        raise ValueError("need at least 2 priors")
    if len(prior_names) != n_priors:
        raise ValueError("prior_names length must match matrix shape")

    tri = np.triu_indices(n_priors, k=1)
    data_vals = data_similarity_matrix[tri]
    perf_vals = performance_similarity_matrix[tri]
    pair_names = [f"{prior_names[i]} <-> {prior_names[j]}" for i, j in zip(tri[0], tri[1])]

    valid = np.isfinite(data_vals) & np.isfinite(perf_vals)
    data_vals = data_vals[valid]
    perf_vals = perf_vals[valid]
    pair_names = [name for name, keep in zip(pair_names, valid) if keep]

    if data_vals.size == 0:
        raise ValueError("no valid prior pairs to plot")

    gap = np.abs(perf_vals - data_vals)
    mismatch_matrix = performance_similarity_matrix - data_similarity_matrix

    mismatch_scale = float(np.nanpercentile(np.abs(mismatch_matrix), 95))
    if not np.isfinite(mismatch_scale) or mismatch_scale <= 0:
        mismatch_scale = float(np.nanmax(np.abs(mismatch_matrix)))
    if not np.isfinite(mismatch_scale) or mismatch_scale <= 0:
        mismatch_scale = 1.0

    mismatch_threshold = float(np.nanpercentile(gap, 75)) if gap.size else 0.0
    if not np.isfinite(mismatch_threshold):
        mismatch_threshold = 0.0

    ranked_pairs = sorted(
        [
            {
                "pair": pair,
                "data_similarity": float(dx),
                "performance_similarity": float(dy),
                "gap": float(dg),
                "signed_gap": float(dy - dx),
                "relationship": (
                    "data_similar_perf_different"
                    if dx >= high_similarity_threshold and dy < high_similarity_threshold
                    else (
                        "data_different_perf_similar"
                        if dx < high_similarity_threshold and dy >= high_similarity_threshold
                        else "agreement"
                    )
                ),
            }
            for pair, dx, dy, dg in zip(pair_names, data_vals, perf_vals, gap)
        ],
        key=lambda row: row["gap"],
        reverse=True,
    )

    top_data_similar_perf_diff = [
        row for row in ranked_pairs
        if row["data_similarity"] >= high_similarity_threshold
        and row["performance_similarity"] < high_similarity_threshold
    ][:top_k]
    top_data_diff_perf_similar = [
        row for row in ranked_pairs
        if row["data_similarity"] < high_similarity_threshold
        and row["performance_similarity"] >= high_similarity_threshold
    ][:top_k]

    return {
        "num_pairs": int(data_vals.size),
        "ranked_pairs": ranked_pairs,
        "ranked_agreements": sorted(ranked_pairs, key=lambda row: row["gap"]),
        "top_data_similar_perf_different": top_data_similar_perf_diff,
        "top_data_different_perf_similar": top_data_diff_perf_similar,
        "high_similarity_threshold": float(high_similarity_threshold),
        "mismatch_threshold": float(mismatch_threshold),
        "mismatch_scale": float(mismatch_scale),
        "mismatch_matrix": mismatch_matrix,
    }


def plot_data_vs_performance_similarity(
    data_similarity_matrix: np.ndarray,
    performance_similarity_matrix: np.ndarray,
    prior_names: list[str],
    output_path: str = "data_vs_performance_similarity.png",
    top_k: int = 10,
):
    """Plot data similarity, performance similarity, and their disagreement side by side."""

    # Scale performance similarity from [-1, 1] to [0, 1] to match data similarity scale
    norm_perf_matrix = (performance_similarity_matrix + 1.0) / 2.0

    stats = compute_data_vs_performance_stats(
        data_similarity_matrix, norm_perf_matrix, prior_names, top_k=top_k,
    )
    ranked_pairs = stats["ranked_pairs"]
    mismatch_matrix = stats["mismatch_matrix"]
    mismatch_scale = stats["mismatch_scale"]
    n_priors = data_similarity_matrix.shape[0]

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.0, 0.9, 0.9],
        width_ratios=[1.0, 1.0, 1.0],
        wspace=0.42,
        hspace=0.48,
    )
    data_ax = fig.add_subplot(gs[0, 0])
    perf_ax = fig.add_subplot(gs[0, 1])
    mismatch_ax = fig.add_subplot(gs[0, 2])
    rank_ax = fig.add_subplot(gs[1, :])
    agreement_ax = fig.add_subplot(gs[2, :])

    data_im = data_ax.imshow(
        data_similarity_matrix,
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    perf_im = perf_ax.imshow(
        norm_perf_matrix,
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )

    mismatch_im = mismatch_ax.imshow(
        mismatch_matrix,
        cmap="coolwarm",
        vmin=-mismatch_scale,
        vmax=mismatch_scale,
        aspect="equal",
    )

    annotation_fontsize = max(6, min(8, int(95 / max(n_priors, 1))))

    def _annotate_heatmap(ax, matrix, text_color, nan_label="—"):
        for i in range(n_priors):
            for j in range(n_priors):
                value = matrix[i, j]
                label = nan_label if not np.isfinite(value) else f"{value:.2f}"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=annotation_fontsize,
                    color=text_color,
                )

    def _format_heatmap(ax, title, show_y_labels: bool = False):
        ax.set_xticks(range(n_priors))
        ax.set_xticklabels(prior_names, rotation=50, ha="right", fontsize=8)
        ax.set_yticks(range(n_priors))
        if show_y_labels:
            ax.set_yticklabels(prior_names, fontsize=8)
            ax.set_ylabel("Prior")
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Prior", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

    _format_heatmap(data_ax, "Data Similarity", show_y_labels=True)
    _format_heatmap(perf_ax, "Performance Similarity")
    _format_heatmap(mismatch_ax, "Performance - Data")

    _annotate_heatmap(data_ax, data_similarity_matrix, "white")
    _annotate_heatmap(perf_ax, norm_perf_matrix, "white")
    _annotate_heatmap(mismatch_ax, mismatch_matrix, "black")

    data_cbar = fig.colorbar(data_im, ax=data_ax, fraction=0.04, pad=0.02)
    data_cbar.set_label("Data similarity", fontsize=9)

    perf_cbar = fig.colorbar(perf_im, ax=perf_ax, fraction=0.04, pad=0.02)
    perf_cbar.set_label("Performance similarity (normalized 0-1)", fontsize=9)

    mismatch_cbar = fig.colorbar(mismatch_im, ax=mismatch_ax, fraction=0.04, pad=0.02)
    mismatch_cbar.set_label("Performance - data", fontsize=9)

    def _plot_pair_gap_bars(ax, title, rows):
        ax.set_title(
            title,
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        if not rows:
            ax.axis("off")
            return

        y_pos = np.arange(len(rows))
        signed_vals = np.array([row["signed_gap"] for row in rows], dtype=float)
        labels = []
        for idx, row in enumerate(rows, start=1):
            short_pair = row["pair"]
            if len(short_pair) > 48:
                short_pair = short_pair[:45] + "..."
            labels.append(f"{idx}. {short_pair}")

        bar_colors = plt.cm.coolwarm(
            0.5 + 0.5 * np.clip(signed_vals / max(mismatch_scale, 1e-9), -1.0, 1.0)
        )
        ax.barh(y_pos, signed_vals, color=bar_colors, alpha=0.9)
        ax.axvline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Performance similarity - data similarity")
        ax.grid(True, alpha=0.25, axis="x")

        max_abs = float(np.max(np.abs(signed_vals))) if signed_vals.size else 1.0
        margin = max(0.04, 0.18 * max_abs)
        ax.set_xlim(
            min(0.0, float(np.min(signed_vals))) - margin,
            max(0.0, float(np.max(signed_vals))) + margin,
        )

        for yi, row, value in zip(y_pos, rows, signed_vals):
            ax.text(
                0.99,
                yi,
                (
                    f"data={row['data_similarity']:.2f}, "
                    f"perf={row['performance_similarity']:.2f}"
                ),
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="right",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
            )

    top_rows = ranked_pairs[: min(top_k, len(ranked_pairs))]
    agreement_rows = stats["ranked_agreements"][: min(top_k, len(ranked_pairs))]
    _plot_pair_gap_bars(rank_ax, "Largest Data/Performance Disagreements", top_rows)
    _plot_pair_gap_bars(agreement_ax, "Largest Data/Performance Agreements", agreement_rows)

    fig.suptitle(
        "Data similarity vs TabArena performance similarity",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    output_path = _resolve_plot_path(output_path)
    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        top=0.88,
        bottom=0.07,
        wspace=0.42,
        hspace=0.5,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📊 Data-vs-performance readability plot saved to: {output_path}")
    plt.close()

    return stats
