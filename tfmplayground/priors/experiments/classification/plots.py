"""Classification-specific visualization functions for prior comparison.

Includes both individual prior sample visualizations and statistical comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA

from tfmplayground.priors.experiments.base_analyzer import (
    compute_prior_similarity_matrix,
)
from tfmplayground.priors.experiments.classification.analyzer import (
    ClassificationDataAnalyzer,
)
from tfmplayground.priors.experiments.utils.general import (
    get_prior_colors,
    apply_plot_style,
    merge_variable_width_features,
)


def _filter_valid_classification_labels(
    labels: np.ndarray, features: Optional[np.ndarray] = None
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Drop masked or invalid class labels before plotting."""
    labels = np.asarray(labels)
    mask = np.isfinite(labels) & (labels >= 0)
    filtered_labels = labels[mask].astype(int, copy=False)
    if features is None:
        return filtered_labels, None
    return filtered_labels, np.asarray(features)[mask]

# individual prior visualizations


def plot_class_samples(
    analyzer: ClassificationDataAnalyzer,
    prior_name: str,
    n_samples: int = 5,
    sample_indices: Optional[list] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot visualizations showing class distributions and feature relationships.

    Left: Bar plot showing class distribution across samples
    Right: Feature space visualization (PCA) colored by class

    Args:
        analyzer: ClassificationDataAnalyzer instance
        prior_name: Name of the prior for title
        n_samples: Number of random samples to use
        sample_indices: Specific sample indices to plot (overrides n_samples)

    Returns:
        Tuple of (figure, axes)
    """
    with apply_plot_style():
        data = analyzer.data
        n_total = len(data["X"])

        if sample_indices is None:
            sample_indices = np.random.choice(
                n_total, min(n_samples, n_total), replace=False
            )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # left: class distribution across samples
        ax = axes[0]
        class_counts = []
        labels = []

        for idx in sample_indices:
            n_points = data["num_datapoints"][idx]
            y, _ = _filter_valid_classification_labels(data["y"][idx, :n_points])
            if y.size == 0:
                continue
            unique, counts = np.unique(y, return_counts=True)
            class_counts.append(dict(zip(unique, counts)))
            labels.append(f"S{idx}")

        if not class_counts:
            raise ValueError(
                f"{prior_name}: no valid class labels found after filtering masked values"
            )

        # get all unique classes across samples
        all_classes = sorted(set().union(*[set(cc.keys()) for cc in class_counts]))

        # create stacked bar chart using only the samples that survived filtering
        positions = np.arange(len(class_counts))
        bottom = np.zeros(len(class_counts))
        colors_list = plt.cm.Set3(np.linspace(0, 1, len(all_classes)))

        for i, cls in enumerate(all_classes):
            counts = [cc.get(cls, 0) for cc in class_counts]
            ax.bar(
                positions,
                counts,
                bottom=bottom,
                label=f"Class {cls}",
                color=colors_list[i],
                alpha=0.8,
            )
            bottom += counts

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Count")
        ax.set_title(f"{prior_name}: Class Distribution per Sample")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

        # right: feature space PCA colored by class
        ax = axes[1]

        # collect data from samples for PCA
        all_features = []
        all_labels = []

        for idx in sample_indices:
            n_points = data["num_datapoints"][idx]
            n_features = data["num_features"][idx]
            x = data["X"][idx, :n_points, :n_features]
            y, x = _filter_valid_classification_labels(data["y"][idx, :n_points], x)
            if y.size == 0:
                continue
            all_features.append(x)
            all_labels.append(y)

        if not all_features:
            raise ValueError(
                f"{prior_name}: no valid class labels found after filtering masked values"
            )

        all_features = merge_variable_width_features(all_features)
        all_labels = np.concatenate(all_labels)

        # PCA projection
        if all_features.shape[1] > 2:
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(all_features)
            explained_var = pca.explained_variance_ratio_
            xlabel = f"PC1 ({explained_var[0]:.1%} var)"
            ylabel = f"PC2 ({explained_var[1]:.1%} var)"
        else:
            features_2d = all_features[:, :2]
            xlabel = "Feature 0"
            ylabel = "Feature 1" if all_features.shape[1] > 1 else "Feature 0"

        scatter = ax.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=all_labels,
            cmap="tab10",
            s=30,
            alpha=0.6,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{prior_name}: Feature Space (colored by class)")
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Class Label")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes


def plot_single_prior_overview(
    analyzer: ClassificationDataAnalyzer, prior_name: str, n_samples: int = 3
) -> Tuple[plt.Figure, np.ndarray]:
    """Comprehensive overview of a single classification prior.

    Shows class distributions, feature distributions, and class separability.

    Args:
        analyzer: ClassificationDataAnalyzer instance
        prior_name: Name of the prior for title
        n_samples: Number of samples to show

    Returns:
        Tuple of (figure, axes)
    """
    with apply_plot_style():
        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(3, 2, hspace=0.50, wspace=0.4)

        color = "tab:blue"
        data = analyzer.data

        # top row: PCA projection colored by class
        ax1 = fig.add_subplot(gs[0, :])
        sample_indices = np.random.choice(
            len(data["X"]), min(n_samples, len(data["X"])), replace=False
        )

        # collect data from multiple samples
        all_features = []
        all_labels = []

        for idx in sample_indices:
            n_points = data["num_datapoints"][idx]
            n_features = data["num_features"][idx]
            x = data["X"][idx, :n_points, :n_features]
            y, x = _filter_valid_classification_labels(data["y"][idx, :n_points], x)
            if y.size == 0:
                continue
            all_features.append(x)
            all_labels.append(y)

        if not all_features:
            raise ValueError(
                f"{prior_name}: no valid class labels found after filtering masked values"
            )

        all_features = merge_variable_width_features(all_features)
        all_labels = np.concatenate(all_labels)

        # PCA projection
        if all_features.shape[1] > 2:
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(all_features)
            explained_var = pca.explained_variance_ratio_
            xlabel = f"PC1 ({explained_var[0]:.1%} var)"
            ylabel = f"PC2 ({explained_var[1]:.1%} var)"
        else:
            features_2d = all_features[:, :2]
            xlabel = "Feature 0"
            ylabel = "Feature 1" if all_features.shape[1] > 1 else "Feature 0"

        scatter = ax1.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=all_labels,
            cmap="tab10",
            s=30,
            alpha=0.6,
        )
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(f"{prior_name}: Feature Space (colored by class)")
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label("Class Label")
        ax1.grid(True, alpha=0.3)

        # middle-left: feature distribution boxplot
        ax2 = fig.add_subplot(gs[1, 0])
        all_features_matrix = analyzer.get_all_features()
        # Filter out NaN values and only show features with substantial data
        feature_data = []
        feature_indices = []
        for i in range(all_features_matrix.shape[1]):
            col_data = all_features_matrix[:, i][~np.isnan(all_features_matrix[:, i])]
            if col_data.size > 0 and np.std(col_data) > 1e-6:  # Has data and non-zero variance
                feature_data.append(col_data)
                feature_indices.append(i)
                if len(feature_data) >= 10:  # Show up to 10 features
                    break
        if not feature_data:
            ax2.text(
                0.5,
                0.5,
                "No non-constant features",
                ha="center",
                va="center",
            )
            ax2.axis("off")
        else:
            bp = ax2.boxplot(feature_data, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax2.set_xlabel("Feature Index")
            ax2.set_ylabel("Feature Value")
            ax2.set_title(f"Feature Distributions ({len(feature_data)} features with data)")
            ax2.set_xticks(range(1, len(feature_data) + 1))
            ax2.set_xticklabels(feature_indices)
            ax2.grid(True, alpha=0.3, axis="y")

        # middle-right: class distribution
        ax3 = fig.add_subplot(gs[1, 1])
        all_labels_flat, _ = _filter_valid_classification_labels(analyzer.get_all_targets())
        if all_labels_flat.size == 0:
            raise ValueError(
                f"{prior_name}: no valid class labels found after filtering masked values"
            )
        unique_classes, class_counts = np.unique(all_labels_flat, return_counts=True)
        class_probs = class_counts / class_counts.sum()

        colors_list = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))
        ax3.bar(
            unique_classes,
            class_probs,
            color=colors_list,
            alpha=0.7,
            edgecolor="black",
        )
        ax3.set_xlabel("Class Label")
        ax3.set_ylabel("Proportion")
        ax3.set_title("Class Distribution (normalized)")
        ax3.set_xticks(unique_classes)
        ax3.grid(True, alpha=0.3, axis="y")

        # bottom-left: class balance across samples
        ax4 = fig.add_subplot(gs[2, 0])

        sample_balances = []
        for i in range(min(20, len(data["X"]))):
            n_points = data["num_datapoints"][i]
            y, _ = _filter_valid_classification_labels(data["y"][i, :n_points])
            if y.size == 0:
                continue
            unique, counts = np.unique(y, return_counts=True)
            if unique.size == 0:
                continue
            # calculate balance as entropy normalized to [0, 1]
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(len(unique))
            balance = entropy / max_entropy if max_entropy > 0 else 1.0
            sample_balances.append(balance)

        if not sample_balances:
            raise ValueError(
                f"{prior_name}: no valid class labels found after filtering masked values"
            )

        ax4.hist(sample_balances, bins=20, color=color, alpha=0.7, edgecolor="black")
        ax4.axvline(
            np.mean(sample_balances),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(sample_balances):.2f}",
        )
        ax4.set_xlabel("Class Balance (normalized entropy)")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Class Balance Distribution Across Samples")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # bottom-right: summary statistics
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis("off")

        target_stats = analyzer.analyze_target_distribution()

        stats_text = f"""
        {prior_name} Summary Statistics
        
        Class Statistics:
        Num Classes: {target_stats['num_classes']:.2f}
        Imbalance Ratio: {target_stats['imbalance_ratio']:.3f}
        Majority Ratio: {target_stats['majority_ratio']:.1%}
        Minority Ratio: {target_stats['minority_ratio']:.1%}
        
        Dataset Info:
        Num Samples: {len(data['X'])}
        Num Features: {data['num_features'][0]}
        Avg Seq Length: {np.mean(data['num_datapoints']):.1f}
        """
        ax5.text(
            0.1,
            0.5,
            stats_text,
            fontsize=11,
            family="monospace",
            verticalalignment="center",
        )

        fig.suptitle(
            f"{prior_name} Prior: Comprehensive Overview",
            fontsize=14,
            fontweight="bold",
        )

        return fig, fig.axes


# statistical comparison visualizations across priors


def plot_class_distributions(
    analyzers: Dict[str, ClassificationDataAnalyzer],
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot class distributions across priors.

    Shows number of classes and class balance.

    Args:
        analyzers: Dict mapping prior names to ClassificationDataAnalyzer instances

    Returns:
        Tuple of (figure, axes)
    """
    with apply_plot_style():
        prior_names = list(analyzers.keys())
        colors = get_prior_colors(prior_names)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # left: number of classes per prior
        ax = axes[0]
        num_classes = []

        for prior_name in prior_names:
            target_stats = analyzers[prior_name].analyze_target_distribution()
            num_classes.append(target_stats["num_classes"])

        positions = np.arange(len(prior_names))
        ax.bar(
            positions, num_classes, color=[colors[p] for p in prior_names], alpha=0.7
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Number of Classes")
        ax.set_title("Number of Classes per Prior")
        ax.grid(True, alpha=0.3, axis="y")

        # right: class balance
        ax = axes[1]
        class_balances = []

        for prior_name in prior_names:
            target_stats = analyzers[prior_name].analyze_target_distribution()
            # Normalize entropy to [0,1] where 1 is perfect balance
            max_entropy = (
                np.log2(target_stats["num_classes"])
                if target_stats["num_classes"] > 1
                else 1.0
            )
            balance = (
                target_stats["entropy_bits"] / max_entropy if max_entropy > 0 else 1.0
            )
            class_balances.append(balance)

        ax.bar(
            positions, class_balances, color=[colors[p] for p in prior_names], alpha=0.7
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Class Balance (normalized entropy)")
        ax.set_title("Class Balance per Prior")
        ax.axhline(
            y=1.0,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Perfect Balance",
        )
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig, axes


def plot_feature_distributions(
    analyzers: Dict[str, ClassificationDataAnalyzer],
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot feature distribution statistics across priors.

    Shows feature means and standard deviations.

    Args:
        analyzers: Dict mapping prior names to ClassificationDataAnalyzer instances

    Returns:
        Tuple of (figure, axes)
    """
    with apply_plot_style():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = get_prior_colors(list(analyzers.keys()))

        prior_names = list(analyzers.keys())
        positions = np.arange(len(prior_names))

        # left: feature means
        ax = axes[0]
        feature_means = []

        for prior_name in prior_names:
            features = analyzers[prior_name].get_all_features()
            feature_means.append(np.nanmean(features))

        ax.bar(
            positions, feature_means, color=[colors[p] for p in prior_names], alpha=0.7
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Mean Feature Value")
        ax.set_title("Average Feature Values")
        ax.grid(True, alpha=0.3, axis="y")

        # right: feature standard deviations
        ax = axes[1]
        feature_stds = []

        for prior_name in prior_names:
            features = analyzers[prior_name].get_all_features()
            feature_stds.append(np.nanstd(features))

        ax.bar(
            positions, feature_stds, color=[colors[p] for p in prior_names], alpha=0.7
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Feature Std Dev")
        ax.set_title("Feature Variability")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig, axes


def plot_class_separability(
    analyzers: Dict[str, ClassificationDataAnalyzer],
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot ANOVA F feature-label association metrics across priors.

    Shows the mean and median F-statistics from per-task feature-label tests.

    Args:
        analyzers: Dict mapping prior names to ClassificationDataAnalyzer instances

    Returns:
        Tuple of (figure, axes)
    """
    with apply_plot_style():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = get_prior_colors(list(analyzers.keys()))

        prior_names = list(analyzers.keys())
        positions = np.arange(len(prior_names))

        # left: mean ANOVA F-statistic
        ax = axes[0]
        f_means = []
        f_medians = []

        for prior_name in prior_names:
            rel_stats = analyzers[prior_name].analyze_target_feature_relationships()
            f_means.append(float(rel_stats.get("f_mean", 0.0)) if rel_stats else 0.0)
            f_medians.append(float(rel_stats.get("f_median", 0.0)) if rel_stats else 0.0)

        ax.bar(
            positions,
            f_means,
            color=[colors[p] for p in prior_names],
            alpha=0.7,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Mean ANOVA F")
        ax.set_title("Feature-Label Association (Mean F)")
        ax.grid(True, alpha=0.3, axis="y")

        # right: median ANOVA F-statistic
        ax = axes[1]
        ax.bar(
            positions,
            f_medians,
            color=[colors[p] for p in prior_names],
            alpha=0.7,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Median ANOVA F")
        ax.set_title("Feature-Label Association (Median F)")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig, axes


def plot_complexity_metrics(
    analyzers: Dict[str, ClassificationDataAnalyzer],
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot complexity characteristics across priors.

    Shows dimensionality, sequence length, and task complexity metrics.

    Args:
        analyzers: Dict mapping prior names to ClassificationDataAnalyzer instances

    Returns:
        Tuple of (figure, axes)
    """
    with apply_plot_style():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        colors = get_prior_colors(list(analyzers.keys()))

        prior_names = list(analyzers.keys())
        positions = np.arange(len(prior_names))

        # top-left: number of features
        ax = axes[0, 0]
        num_features = []

        for prior_name in prior_names:
            basic_stats = analyzers[prior_name].get_basic_statistics()
            num_features.append(basic_stats["num_features"]["mean"])

        ax.bar(
            positions, num_features, color=[colors[p] for p in prior_names], alpha=0.7
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Number of Features")
        ax.set_title("Dimensionality")
        ax.grid(True, alpha=0.3, axis="y")

        # top-right: sequence length
        ax = axes[0, 1]
        seq_lengths = []

        for prior_name in prior_names:
            basic_stats = analyzers[prior_name].get_basic_statistics()
            seq_lengths.append(basic_stats["seq_lengths"]["mean"])

        ax.bar(
            positions, seq_lengths, color=[colors[p] for p in prior_names], alpha=0.7
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Sequence Length")
        ax.set_title("Average Sequence Length")
        ax.grid(True, alpha=0.3, axis="y")

        # bottom-left: number of classes
        ax = axes[1, 0]
        num_classes = []

        for prior_name in prior_names:
            target_stats = analyzers[prior_name].analyze_target_distribution()
            num_classes.append(target_stats["num_classes"])

        ax.bar(
            positions, num_classes, color=[colors[p] for p in prior_names], alpha=0.7
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Number of Classes")
        ax.set_title("Task Complexity (Num Classes)")
        ax.grid(True, alpha=0.3, axis="y")

        # bottom-right: samples per class
        ax = axes[1, 1]
        samples_per_class = []

        for prior_name in prior_names:
            basic_stats = analyzers[prior_name].get_basic_statistics()
            target_stats = analyzers[prior_name].analyze_target_distribution()
            avg_seq_len = basic_stats["seq_lengths"]["mean"]
            num_cls = target_stats["num_classes"]
            samples_per_class.append(avg_seq_len / num_cls if num_cls > 0 else 0)

        ax.bar(
            positions,
            samples_per_class,
            color=[colors[p] for p in prior_names],
            alpha=0.7,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Samples per Class")
        ax.set_title("Average Samples per Class")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig, axes


def plot_feature_redundancy(
    analyzers: Dict[str, ClassificationDataAnalyzer],
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot feature redundancy analysis across priors.

    Shows inter-feature correlations and redundancy metrics.

    Args:
        analyzers: Dict mapping prior names to ClassificationDataAnalyzer instances

    Returns:
        Tuple of (figure, axes)
    """
    with apply_plot_style():
        n_priors = len(analyzers)
        total_plots = n_priors + 1  # one heatmap per prior + one summary plot
        
        # 3 plots per row max
        n_cols = min(total_plots, 3)
        n_rows = (total_plots + n_cols - 1) // n_cols  # ceil division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        
        # Flatten axes for easy indexing
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = list(axes) if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        colors = get_prior_colors(list(analyzers.keys()))

        # heatmaps for each prior
        for idx, (prior_name, analyzer) in enumerate(analyzers.items()):
            ax = axes[idx]
            features = analyzer.get_all_features()

            # compute correlation matrix
            if features.shape[1] > 1:
                masked_features = np.ma.masked_invalid(features)
                corr_matrix = np.ma.corrcoef(masked_features, rowvar=False).filled(0.0)
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                np.fill_diagonal(corr_matrix, 1.0)
            else:
                corr_matrix = np.array([[1.0]])

            im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_title(f"{prior_name}\nFeature Correlations")
            ax.set_xlabel("Feature Index")
            ax.set_ylabel("Feature Index")

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # summary plot: mean absolute correlation
        ax = axes[n_priors]  # Last used plot index
        prior_names = list(analyzers.keys())
        positions = np.arange(len(prior_names))
        mean_corrs = []

        for prior_name in prior_names:
            red_stats = analyzers[prior_name].analyze_feature_redundancy()
            mean_corrs.append(float(red_stats.get("mean_abs_correlation", 0.0)))

        ax.bar(positions, mean_corrs, color=[colors[p] for p in prior_names], alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Mean |Correlation|")
        ax.set_title("Feature Redundancy")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis="y")

        # hide any unused subplots from the grid
        for idx in range(total_plots, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        return fig, axes


def plot_prior_similarity(
    analyzers: Dict[str, ClassificationDataAnalyzer],
    annotate: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Compare priors based on standardized summary-statistic proximity.

    Args:
        analyzers: Dict mapping prior names to ClassificationDataAnalyzer instances
        annotate: Whether to print numeric similarity values in heatmap cells.

    Returns:
        Tuple of (figure, axes)
    """

    with apply_plot_style():
        prior_names, sim_matrix = compute_prior_similarity_matrix(analyzers)
        n_priors = len(prior_names)

        # heatmap
        fig, ax = plt.subplots(figsize=(6 + 0.5 * n_priors, 5))
        im = ax.imshow(sim_matrix, vmin=0.0, vmax=1.0, cmap="coolwarm")

        ax.set_xticks(np.arange(n_priors))
        ax.set_yticks(np.arange(n_priors))
        ax.set_xticklabels(prior_names, rotation=45, ha="right")
        ax.set_yticklabels(prior_names)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Prior")
        ax.set_title("Prior Similarity (distance-based meta-feature similarity)")

        if annotate:
            for i in range(n_priors):
                for j in range(n_priors):
                    ax.text(
                        j,
                        i,
                        f"{sim_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color="white",
                    )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Similarity")

        plt.tight_layout()
        return fig, ax
