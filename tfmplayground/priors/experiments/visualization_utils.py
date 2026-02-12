"""Visualization utilities for model comparison and evaluation."""

import os

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
        metric_history = cb.score_history if hasattr(cb, "score_history") else cb.accuracy_history
        xs, ys = [], []
        for e, m in zip(cb.epoch_history, metric_history):
            if m is None:
                continue
            xs.append(e)
            ys.append(m)
        ax2.plot(xs, ys, label=name, linewidth=2)

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel(metric_name, fontsize=12)
    ax2.set_title(f"Validation {metric_name} Comparison", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Legends outside
    ax1.legend(fontsize=9, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
    ax2.legend(fontsize=9, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Comparison plot saved to: {save_path}")

    try:
        plt.show()
    except:
        pass

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
    model, X_train, X_test, y_train, y_test, ax, title="Model", resolution=100, device=None
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
            xx, yy, proba_c, levels=[0.5], colors=[colors[c]], linewidths=2, linestyles="solid"
        )

    cmap_bold = ListedColormap(["#CC0000", "#0000CC"])

    ax.scatter(
        X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold,
        edgecolors="white", s=30, linewidth=1, marker="o", alpha=0.8
    )
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold,
        edgecolors="black", s=25, linewidth=1, marker="s", alpha=0.6
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
    ax.plot(X_plot, y_plot, 'b-', linewidth=2, label='Prediction', alpha=0.8)
    
    # Plot training points
    ax.scatter(X_train, y_train, c='green', s=30, alpha=0.6, edgecolors='white', 
               linewidth=1, label='Train', marker='o')
    
    # Plot test points
    ax.scatter(X_test, y_test, c='red', s=25, alpha=0.6, edgecolors='black',
               linewidth=1, label='Test', marker='s')
    
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.set_title(f"{title}\nR²: {r2:.3f}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc='best')
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
    ax.plot(X_plot, y_plot, 'b-', linewidth=2, label='Prediction', alpha=0.8)
    
    # Plot training points
    ax.scatter(X_train, y_train, c='green', s=30, alpha=0.6, edgecolors='white',
               linewidth=1, label='Train', marker='o')
    
    # Plot test points
    ax.scatter(X_test, y_test, c='red', s=25, alpha=0.6, edgecolors='black',
               linewidth=1, label='Test', marker='s')
    
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.set_title(f"{title}\nR²: {r2:.3f}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    
    return r2


def plot_all_regression_predictions(
    run_records, datasets=["sinusoidal", "linear", "step"],
    n_samples=100, noise=0.1, seed=42, output_path="regression_predictions_comparison.png"
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
                baseline_info["model"], X_train, X_test, y_train, y_test, ax, title=title
            )
            
            if dataset_idx == 0 and baseline_idx == 0:
                ax.text(
                    -0.3, 0.5, dataset_name.capitalize(),
                    transform=ax.transAxes, fontsize=12, fontweight="bold",
                    va="center", ha="right", rotation=90
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
                    -0.3, 0.5, dataset_name.capitalize(),
                    transform=ax.transAxes, fontsize=12, fontweight="bold",
                    va="center", ha="right", rotation=90
                )
    
    plt.suptitle(
        "Regression Predictions on Toy Datasets", fontsize=14, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Regression predictions plot saved to: {output_path}")
    
    try:
        plt.show()
    except:
        pass
    
    plt.close()


def plot_all_decision_boundaries(
    run_records, datasets=["moons", "circles"], n_samples=200, noise=0.2, seed=42, output_path="decision_boundaries_comparison.png"
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
                baseline_info["model"], X_train, X_test, y_train, y_test, ax, title=title
            )

            if dataset_idx == 0 and baseline_idx == 0:
                ax.text(
                    -0.3, 0.5, dataset_name.capitalize(),
                    transform=ax.transAxes, fontsize=12, fontweight="bold",
                    va="center", ha="right", rotation=90
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
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Decision boundaries plot saved to: {output_path}")

    try:
        plt.show()
    except:
        pass

    plt.close()
