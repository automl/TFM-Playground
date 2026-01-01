"""Classification-specific visualization functions for prior comparison.

Includes both individual prior sample visualizations and statistical comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.decomposition import PCA

from tfmplayground.priors.experiments.classification.analyzer import (
    ClassificationDataAnalyzer,
)
from tfmplayground.priors.experiments.utils import (
    get_prior_colors,
    apply_plot_style,
)

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
        color = "tab:blue"

        # left: class distribution across samples
        ax = axes[0]
        class_counts = []
        labels = []

        for idx in sample_indices:
            n_points = data["num_datapoints"][idx]
            y = data["y"][idx, :n_points].astype(int)
            unique, counts = np.unique(y, return_counts=True)
            class_counts.append(dict(zip(unique, counts)))
            labels.append(f"S{idx}")

        # get all unique classes across samples
        all_classes = sorted(set().union(*[set(cc.keys()) for cc in class_counts]))

        # create stacked bar chart
        bottom = np.zeros(len(sample_indices))
        colors_list = plt.cm.Set3(np.linspace(0, 1, len(all_classes)))

        for i, cls in enumerate(all_classes):
            counts = [cc.get(cls, 0) for cc in class_counts]
            ax.bar(
                range(len(sample_indices)),
                counts,
                bottom=bottom,
                label=f"Class {cls}",
                color=colors_list[i],
                alpha=0.8,
            )
            bottom += counts

        ax.set_xticks(range(len(sample_indices)))
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
            y = data["y"][idx, :n_points].astype(int)
            all_features.append(x)
            all_labels.append(y)

        all_features = np.vstack(all_features)
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
            y = data["y"][idx, :n_points].astype(int)
            all_features.append(x)
            all_labels.append(y)

        all_features = np.vstack(all_features)
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
        n_features_to_show = min(10, all_features_matrix.shape[1])
        feature_data = [all_features_matrix[:, i] for i in range(n_features_to_show)]
        bp = ax2.boxplot(feature_data, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_xlabel("Feature Index")
        ax2.set_ylabel("Feature Value")
        ax2.set_title(f"Feature Distributions (first {n_features_to_show} features)")
        ax2.grid(True, alpha=0.3, axis="y")

        # middle-right: class distribution
        ax3 = fig.add_subplot(gs[1, 1])
        all_labels_flat = analyzer.get_all_targets().astype(int)
        unique_classes, class_counts = np.unique(all_labels_flat, return_counts=True)

        colors_list = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))
        ax3.bar(
            unique_classes,
            class_counts,
            color=colors_list,
            alpha=0.7,
            edgecolor="black",
        )
        ax3.set_xlabel("Class Label")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Class Distribution")
        ax3.set_xticks(unique_classes)
        ax3.grid(True, alpha=0.3, axis="y")

        # bottom-left: class balance across samples
        ax4 = fig.add_subplot(gs[2, 0])

        sample_balances = []
        for i in range(min(20, len(data["X"]))):
            n_points = data["num_datapoints"][i]
            y = data["y"][i, :n_points].astype(int)
            unique, counts = np.unique(y, return_counts=True)
            # calculate balance as entropy normalized to [0, 1]
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(len(unique))
            balance = entropy / max_entropy if max_entropy > 0 else 1.0
            sample_balances.append(balance)

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
        Num Classes: {target_stats['num_classes']}
        Imbalance Ratio: {target_stats['imbalance_ratio']:.3f}
        Majority Class: {target_stats['majority_class']} ({target_stats['majority_ratio']:.1%})
        Minority Class: {target_stats['minority_class']} ({target_stats['minority_ratio']:.1%})
        
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
            feature_means.append(np.mean(features))

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
            feature_stds.append(np.std(features))

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
    """Plot class separability metrics across priors.

    Shows Fisher discriminant ratio and between/within class variance ratio.

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

        # left: Fisher discriminant ratio distribution (using F-statistics as proxy)
        ax = axes[0]
        fisher_ratios = []

        for prior_name in prior_names:
            # Use ANOVA F-statistics as a proxy for separability
            rel_stats = analyzers[prior_name].analyze_target_feature_relationships()
            # Use F-mean as representative value, create a small distribution for visualization
            if rel_stats and "f_mean" in rel_stats:
                f_mean = rel_stats["f_mean"]
                f_std = rel_stats.get("f_std", f_mean * 0.2)
                # Generate approximate samples
                fisher_ratios.append(
                    [f_mean] * 10
                )  # Use mean value for consistent visualization
            else:
                fisher_ratios.append([0.0])

        vp = ax.violinplot(
            fisher_ratios,
            positions=positions,
            widths=0.7,
            showmeans=True,
            showmedians=True,
        )

        for i, pc in enumerate(vp["bodies"]):
            pc.set_facecolor(colors[prior_names[i]])
            pc.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Fisher Discriminant Ratio")
        ax.set_title("Class Separability (Fisher Ratio)")
        ax.grid(True, alpha=0.3, axis="y")

        # right: mean separability
        ax = axes[1]
        mean_separability = [np.mean(fr) if len(fr) > 0 else 0 for fr in fisher_ratios]

        ax.bar(
            positions,
            mean_separability,
            color=[colors[p] for p in prior_names],
            alpha=0.7,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Mean Fisher Ratio")
        ax.set_title("Average Class Separability")
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
        fig, axes = plt.subplots(1, n_priors + 1, figsize=(4 * (n_priors + 1), 4))
        colors = get_prior_colors(list(analyzers.keys()))

        if n_priors == 1:
            axes = [axes] if not isinstance(axes, np.ndarray) else axes

        # heatmaps for each prior
        for idx, (prior_name, analyzer) in enumerate(analyzers.items()):
            ax = axes[idx]
            features = analyzer.get_all_features()

            # compute correlation matrix
            if features.shape[1] > 1:
                corr_matrix = np.corrcoef(features.T)
            else:
                corr_matrix = np.array([[1.0]])

            im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_title(f"{prior_name}\nFeature Correlations")
            ax.set_xlabel("Feature Index")
            ax.set_ylabel("Feature Index")

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # summary plot: mean absolute correlation
        ax = axes[-1]
        prior_names = list(analyzers.keys())
        positions = np.arange(len(prior_names))
        mean_corrs = []

        for prior_name in prior_names:
            features = analyzers[prior_name].get_all_features()
            if features.shape[1] > 1:
                corr_matrix = np.corrcoef(features.T)
                # exclude diagonal
                mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                mean_corrs.append(np.abs(corr_matrix[mask]).mean())
            else:
                mean_corrs.append(0.0)

        ax.bar(positions, mean_corrs, color=[colors[p] for p in prior_names], alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Mean |Correlation|")
        ax.set_title("Feature Redundancy")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig, axes


def prior_summary_vector(
    analyzer: ClassificationDataAnalyzer,
) -> Tuple[np.ndarray, List[str]]:
    """Build a fixed-length numeric summary vector for one classification prior.

    Returns:
        (vector, metric_names) for consistent ordering
    """
    # get statistics from analyzer
    basic = analyzer.get_basic_statistics()
    target_stats = analyzer.analyze_target_distribution()
    rel_stats = analyzer.analyze_target_feature_relationships()

    # extract scalar metrics
    features = analyzer.get_all_features()

    metrics: Dict[str, float] = {
        # dimensionality
        "dim_num_features_mean": float(basic["num_features"].get("mean", 0.0)),
        "dim_num_features_std": float(basic["num_features"].get("std", 0.0)),
        "dim_seq_len_mean": float(basic["seq_lengths"].get("mean", 0.0)),
        "dim_seq_len_std": float(basic["seq_lengths"].get("std", 0.0)),
        "dim_eval_pos_mean": float(basic["eval_positions"].get("mean", 0.0)),
        # class distribution
        "class_num_classes": float(target_stats.get("num_classes", 0.0)),
        "class_imbalance_ratio": float(target_stats.get("imbalance_ratio", 0.0)),
        "class_entropy": float(target_stats.get("entropy_bits", 0.0)),
        # feature statistics
        "feat_mean": float(np.mean(features)),
        "feat_std": float(np.std(features)),
        # separability (using F-statistics)
        "sep_mean_f": float(rel_stats.get("f_mean", 0.0)),
        "sep_median_f": float(rel_stats.get("f_median", 0.0)),
    }

    metric_names: List[str] = list(metrics.keys())
    vec = np.array([metrics[name] for name in metric_names], dtype=float)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    return vec, metric_names


def plot_prior_similarity(
    analyzers: Dict[str, ClassificationDataAnalyzer],
) -> Tuple[plt.Figure, plt.Axes]:
    """Compare priors based on summary statistics.

    Builds a summary vector per prior and plots correlation heatmap.

    Args:
        analyzers: Dict mapping prior names to ClassificationDataAnalyzer instances

    Returns:
        Tuple of (figure, axes)
    """
    with apply_plot_style():
        prior_names = list(analyzers.keys())
        n_priors = len(prior_names)

        # build summary matrix
        summary_vectors = []
        for name in prior_names:
            vec, _ = prior_summary_vector(analyzers[name])
            summary_vectors.append(vec)

        summary_matrix = np.vstack(summary_vectors)

        # standardize for scale invariance
        mean = summary_matrix.mean(axis=0, keepdims=True)
        std = summary_matrix.std(axis=0, keepdims=True) + 1e-8
        summary_z = (summary_matrix - mean) / std

        # compute similarity via correlation
        sim_matrix = np.corrcoef(summary_z)
        sim_matrix = np.nan_to_num(sim_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # heatmap
        fig, ax = plt.subplots(figsize=(6 + 0.5 * n_priors, 5))
        im = ax.imshow(sim_matrix, vmin=-1, vmax=1, cmap="RdBu_r")

        ax.set_xticks(np.arange(n_priors))
        ax.set_yticks(np.arange(n_priors))
        ax.set_xticklabels(prior_names, rotation=45, ha="right")
        ax.set_yticklabels(prior_names)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Prior")
        ax.set_title("Prior Similarity (correlation of summary statistics)")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation")

        plt.tight_layout()
        return fig, ax
