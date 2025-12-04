"""Regression-specific visualization functions for prior comparison.

Includes both individual prior sample visualizations and statistical comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.decomposition import PCA

from tfmplayground.priors.experiments.regression.analyzer import RegressionDataAnalyzer
from tfmplayground.priors.experiments.utils import (
    get_prior_colors, 
    apply_plot_style,
)

# individual prior visualizations

def plot_function_samples(analyzer: RegressionDataAnalyzer, 
                         prior_name: str,
                         n_samples: int = 5,
                         sample_indices: Optional[list] = None) -> Tuple[plt.Figure, np.ndarray]:
    """Plot visualizations showing the relationship between features and targets.
    
    Left: Violin plot showing target distribution across samples
    Right: Scatter matrix of first 3 features colored by target value
    
    Args:
        analyzer: RegressionDataAnalyzer instance
        prior_name: Name of the prior for title
        n_samples: Number of random samples to use for scatter matrix
        sample_indices: Specific sample indices to plot (overrides n_samples)
        
    Returns:
        Tuple of (figure, axes)
    """
    with apply_plot_style():
        data = analyzer.data
        n_total = len(data['X'])
        
        if sample_indices is None:
            sample_indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        color = "tab:blue"

        # left: target distribution violin plot across samples
        ax = axes[0]
        target_distributions = []
        labels = []
        for idx in sample_indices:
            n_points = data['num_datapoints'][idx]
            y = data['y'][idx, :n_points]
            target_distributions.append(y)
            labels.append(f'S{idx}')
        
        parts = ax.violinplot(target_distributions, positions=range(len(sample_indices)),
                             widths=0.7, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(sample_indices)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Target Value')
        ax.set_title(f'{prior_name}: Target Distribution per Sample')
        ax.grid(True, alpha=0.3, axis='y')
        
        # right: feature correlation heatmap (mean correlation across samples)
        ax = axes[1]
        n_features_to_show = min(10, data['num_features'][0])
        
        # compute mean correlation matrix across samples
        correlations = []
        for idx in sample_indices:
            n_points = data['num_datapoints'][idx]
            n_features = data['num_features'][idx]
            x = data['X'][idx, :n_points, :min(n_features_to_show, n_features)]
            y = data['y'][idx, :n_points]
            
            # combine features and target for correlation
            combined = np.column_stack([x, y.reshape(-1, 1)])
            corr = np.corrcoef(combined.T)
            correlations.append(corr)
        
        mean_corr = np.mean(correlations, axis=0)
        
        im = ax.imshow(mean_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(n_features_to_show + 1))
        ax.set_yticks(range(n_features_to_show + 1))
        labels = [f'F{i}' for i in range(n_features_to_show)] + ['Target']
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)
        ax.set_title(f'{prior_name}: Feature-Target Correlations')
        
        plt.colorbar(im, ax=ax, label='Correlation')
        
        plt.tight_layout()
        return fig, axes


def plot_2d_heatmap(analyzer: RegressionDataAnalyzer,
                    prior_name: str,
                    sample_idx: int = 0,
                    feature_indices: Tuple[int, int] = (0, 1)) -> Tuple[plt.Figure, plt.Axes]:
    """Plot 2D heatmap/contour for a single sample with 2+ features.
    
    Args:
        analyzer: RegressionDataAnalyzer instance
        prior_name: Name of the prior for title
        sample_idx: Which sample to visualize
        feature_indices: Which two features to plot (x_axis, y_axis)
        
    Returns:
        Tuple of (figure, axes)
    """
    with apply_plot_style():
        data = analyzer.data
        n_points = data['num_datapoints'][sample_idx]
        n_features = data['num_features'][sample_idx]
        x = data['X'][sample_idx, :n_points, :n_features]
        y = data['y'][sample_idx, :n_points]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        feat_x_idx, feat_y_idx = feature_indices
        scatter = ax.scatter(x[:, feat_x_idx], x[:, feat_y_idx], 
                           c=y, cmap='viridis', s=50, alpha=0.7)
        
        ax.set_xlabel(f'Feature {feat_x_idx}')
        ax.set_ylabel(f'Feature {feat_y_idx}')
        ax.set_title(f'{prior_name}: 2D Feature Space (Sample {sample_idx})')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Target Value')
        
        plt.tight_layout()
        return fig, ax


def plot_single_prior_overview(analyzer: RegressionDataAnalyzer,
                               prior_name: str,
                               n_samples: int = 3) -> Tuple[plt.Figure, np.ndarray]:
    """Comprehensive overview of a single regression prior.
    
    Shows function samples, feature distributions, and target distributions.
    
    Args:
        analyzer: RegressionDataAnalyzer instance
        prior_name: Name of the prior for title
        n_samples: Number of function samples to show
        
    Returns:
        Tuple of (figure, axes)
    """

    with apply_plot_style():
        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(3, 2, hspace=0.50, wspace=0.4)
        
        color = "tab:blue"
        data = analyzer.data
        
        # top row: PCA projection of features colored by target value
        ax1 = fig.add_subplot(gs[0, :])
        sample_indices = np.random.choice(len(data['X']), min(n_samples, len(data['X'])), replace=False)
        
        # collect data from multiple samples for PCA
        all_features = []
        all_targets = []
        for idx in sample_indices:
            n_points = data['num_datapoints'][idx]
            n_features = data['num_features'][idx]
            x = data['X'][idx, :n_points, :n_features]
            y = data['y'][idx, :n_points]
            all_features.append(x)
            all_targets.append(y)
        
        all_features = np.vstack(all_features)
        all_targets = np.concatenate(all_targets)
        
        # simple PCA-like projection: use first 2 principal components
        if all_features.shape[1] > 2:
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(all_features)
            explained_var = pca.explained_variance_ratio_
            xlabel = f'PC1 ({explained_var[0]:.1%} var)'
            ylabel = f'PC2 ({explained_var[1]:.1%} var)'
        else:
            features_2d = all_features[:, :2]
            xlabel = 'Feature 0'
            ylabel = 'Feature 1' if all_features.shape[1] > 1 else 'Feature 0'
        
        scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                             c=all_targets, cmap='viridis', s=30, alpha=0.6)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(f'{prior_name}: Feature Space (colored by target)')
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Target Value')

        ax1.grid(True, alpha=0.3)
        
        # middle-left: box plot of first 10 features to show feature-wise distributions
        ax2 = fig.add_subplot(gs[1, 0])
        all_features_matrix = analyzer.get_all_features()
        n_features_to_show = min(10, all_features_matrix.shape[1])
        feature_data = [all_features_matrix[:, i] for i in range(n_features_to_show)]
        bp = ax2.boxplot(feature_data, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Feature Value')
        ax2.set_title(f'Feature Distributions (first {n_features_to_show} features)')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # middle-right: target value distribution
        ax3 = fig.add_subplot(gs[1, 1])
        all_targets = analyzer.get_all_targets().flatten()
        ax3.hist(all_targets, bins=50, color=color, alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(all_targets), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_targets):.2f}')
        ax3.axvline(np.median(all_targets), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(all_targets):.2f}')
        ax3.set_xlabel('Target Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Target Value Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # bottom-left: feature-target scatter (most predictive feature)
        ax4 = fig.add_subplot(gs[2, 0])

        # find the "best" feature = highest (pearson correlation) with target
        all_features_matrix = analyzer.get_all_features()          # shape (N, D)
        all_targets_vec = analyzer.get_all_targets().flatten()     # shape (N,)

        n_features_total = all_features_matrix.shape[1]
        corrs = []
        for j in range(n_features_total):
            xj = all_features_matrix[:, j]
            # avoid NaNs if feature or target is 'almost' constant
            if np.std(xj) < 1e-12:
                corrs.append(0.0)
            else:
                corr = np.corrcoef(xj, all_targets_vec)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                # use absolute strength
                corrs.append(abs(corr))

        best_feat_idx = int(np.argmax(corrs))

        # collect (best feature, target) pairs from min 10 samples
        sample_x = []
        sample_y = []
        for i in range(min(10, len(data['X']))):
            n_points = data['num_datapoints'][i]
            n_features = data['num_features'][i]
            x = data['X'][i, :n_points, :n_features]
            y = data['y'][i, :n_points]

            if x.shape[1] > best_feat_idx:
                sample_x.extend(x[:, best_feat_idx])
                sample_y.extend(y)

        ax4.scatter(sample_x, sample_y, alpha=0.5, s=20, color=color)
        ax4.set_xlabel(f'Feature {best_feat_idx}')
        ax4.set_ylabel('Target')
        ax4.set_title(f'Most Correlated Feature vs Target (Feature {best_feat_idx})')
        ax4.grid(True, alpha=0.3)
        
        # bottom-right: summary statistics
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        stats_text = f"""
        {prior_name} Summary Statistics
        
        Target Statistics:
        Mean: {np.mean(all_targets):.3f}
        Std: {np.std(all_targets):.3f}
        Min: {np.min(all_targets):.3f}
        Max: {np.max(all_targets):.3f}
        Skewness: {stats.skew(all_targets):.3f}
        
        Feature Statistics:
        Mean: {np.mean(all_features):.3f}
        Std: {np.std(all_features):.3f}
        
        Dataset Info:
        Num Samples: {len(data['X'])}
        Num Features: {data['num_features'][0]}
        Avg Seq Length: {np.mean(data['num_datapoints']):.1f}
        """
        ax5.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', verticalalignment='center')
        
        fig.suptitle(f'{prior_name} Prior: Comprehensive Overview', fontsize=14, fontweight='bold')
        
        return fig, fig.axes


# statistical comparison visualizations across priors

def plot_target_distributions(analyzers: Dict[str, RegressionDataAnalyzer]) -> Tuple[plt.Figure, np.ndarray]:
    """Plot target value distributions across priors.
    
    Creates ridge-style KDE curves and box plots.
    
    Args:
        analyzers: Dict mapping prior names to RegressionDataAnalyzer instances
        
    Returns:
        Tuple of (figure, axes)
    """
    with apply_plot_style():
        prior_names = list(analyzers.keys())
        colors = get_prior_colors(prior_names)

        # make it a bit taller when there are many priors
        fig_height = 5.0 + 0.6 * len(prior_names)
        fig, axes = plt.subplots(1, 2, figsize=(12, fig_height))
        
        # left: ridge-style KDEs instead of fully overlaid histograms
        ax = axes[0]

        # collect all targets once
        all_targets_per_prior = {
            name: analyzers[name].get_all_targets().flatten()
            for name in prior_names
        }

        # common x-range for all priors
        x_min = min(t.min() for t in all_targets_per_prior.values())
        x_max = max(t.max() for t in all_targets_per_prior.values())
        x_range = np.linspace(x_min, x_max, 300)

        # compute KDEs and find global max density for normalization
        kde_values = {}
        max_height = 0.0
        for name in prior_names:
            targets = all_targets_per_prior[name]
            kde = stats.gaussian_kde(targets)
            dens = kde(x_range)
            kde_values[name] = dens
            max_height = max(max_height, dens.max())

        offset = 1.1  # vertical spacing between ridges

        yticks = []
        ylabels = []

        # draw from top to bottom by iterating in reversed order
        for row_idx, name in enumerate(reversed(prior_names)):
            base_color = colors[name]
            dens = kde_values[name] / (max_height + 1e-12)  # normalize
            y0 = row_idx * offset

            # filled ridge
            ax.fill_between(
                x_range,
                y0,
                y0 + dens,
                color=base_color,
                alpha=0.6,
            )
            # outline
            ax.plot(x_range, y0 + dens, color=base_color, linewidth=1.5)
            yticks.append(y0 + 0.5 * dens.max())
            ylabels.append(name)

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

        ax.set_xlabel('Target Value')
        ax.set_ylabel('Prior')
        ax.set_title('Target Distributions')
        ax.grid(True, axis='x', alpha=0.3)

        # right: box plots for target value ranges
        ax = axes[1]
        data = [all_targets_per_prior[name] for name in prior_names]
        positions = np.arange(len(prior_names))
        bp = ax.boxplot(
            data,
            positions=positions,
            labels=prior_names,
            patch_artist=True,
            widths=0.6,
        )
        
        # color boxes and make median a darker shade of the same color
        for box, median, prior_name in zip(bp['boxes'], bp['medians'], prior_names):
            base_hex = colors[prior_name]
            base_rgb = mpl.colors.to_rgb(base_hex)
            darker_rgb = tuple(c * 0.6 for c in base_rgb) 

            box.set_facecolor(base_rgb)
            box.set_alpha(0.7)

            median.set_color(darker_rgb)
            median.set_linewidth(2)

        ax.set_xlabel('Prior')
        ax.set_ylabel('Target Value')
        ax.set_title('Target Value Ranges')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig, axes
    

def plot_feature_distributions(analyzers: Dict[str, RegressionDataAnalyzer]) -> Tuple[plt.Figure, np.ndarray]:
    """Plot feature-target relationship strength across priors.
    
    Shows how strongly features correlate with targets in each prior.
    
    Args:
        analyzers: Dict mapping prior names to RegressionDataAnalyzer instances
        
    Returns:
        Tuple of (figure, axes)
    """

    with apply_plot_style():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = get_prior_colors(list(analyzers.keys()))

        prior_names = list(analyzers.keys())
        x_pos = np.arange(len(prior_names))

        # left: distribution of absolute correlations
        ax = axes[0]
        all_abs_corrs = []

        for prior_name in prior_names:
            rel_stats = analyzers[prior_name].analyze_target_feature_relationships()
            pearson_corrs = np.asarray(rel_stats["pearson_corrs"])   # (sample, feature) correlations
            all_abs_corrs.append(np.abs(pearson_corrs))              # we care about strength, not sign

        vp = ax.violinplot(
            all_abs_corrs,
            positions=x_pos,
            widths=0.7,
            showmeans=True,
            showmedians=True,
        )

        for i, pc in enumerate(vp["bodies"]):
            pc.set_facecolor(colors[prior_names[i]])
            pc.set_alpha(0.7)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Absolute Correlation")
        ax.set_title("Feature-Target Correlation Strength (|Pearson|)")
        ax.grid(True, alpha=0.3, axis="y")

        # right: mean correlation strength
        ax = axes[1]
        mean_corrs = [corrs.mean() for corrs in all_abs_corrs]
        std_corrs = [corrs.std() for corrs in all_abs_corrs]

        ax.bar(
            x_pos,
            mean_corrs,
            yerr=std_corrs,
            color=[colors[p] for p in prior_names],
            alpha=0.7,
            capsize=5,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Mean Absolute Correlation")
        ax.set_title("Average Feature Predictiveness (|Pearson|)")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig, axes
    

def plot_correlations(analyzers: Dict[str, RegressionDataAnalyzer]) -> Tuple[plt.Figure, np.ndarray]:
    """Plot feature-target correlations across priors (Pearson & Spearman)."""

    with apply_plot_style():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = get_prior_colors(list(analyzers.keys()))
        
        prior_names = list(analyzers.keys())
        positions = np.arange(len(prior_names))
        
        # left: pearson correlations
        ax = axes[0]
        data_pearson = []
        
        for prior_name in prior_names:
            rel_stats = analyzers[prior_name].analyze_target_feature_relationships()
            pearson_corrs = np.asarray(rel_stats["pearson_corrs"])
            data_pearson.append(pearson_corrs)
        
        vp = ax.violinplot(
            data_pearson,
            positions=positions,
            widths=0.7,
            showmeans=True,
            showmedians=True,
        )
        for i, pc in enumerate(vp["bodies"]):
            pc.set_facecolor(colors[prior_names[i]])
            pc.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45, ha="right")
        ax.set_xlabel("Prior")
        ax.set_ylabel("Pearson Correlation")
        ax.set_title("Feature-Target Pearson Correlations")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")
        
        # right: spearman correlations
        ax = axes[1]
        data_spearman = []
        
        for prior_name in prior_names:
            rel_stats = analyzers[prior_name].analyze_target_feature_relationships()
            spearman_corrs = np.asarray(rel_stats["spearman_corrs"])
            data_spearman.append(spearman_corrs)
        
        vp = ax.violinplot(
            data_spearman,
            positions=positions,
            widths=0.7,
            showmeans=True,
            showmedians=True,
        )
        for i, pc in enumerate(vp["bodies"]):
            pc.set_facecolor(colors[prior_names[i]])
            pc.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45, ha="right")
        ax.set_xlabel("Prior")
        ax.set_ylabel("Spearman Correlation")
        ax.set_title("Feature-Target Spearman Correlations")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        return fig, axes


def plot_mutual_info(analyzers: Dict[str, RegressionDataAnalyzer]) -> Tuple[plt.Figure, np.ndarray]:
    """Plot mutual information between features and targets.
    
    Shows MI distributions and mean values.
    
    Args:
        analyzers: Dict mapping prior names to RegressionDataAnalyzer instances
        
    Returns:
        Tuple of (figure, axes)
    """
    with apply_plot_style():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = get_prior_colors(list(analyzers.keys()))
        
        prior_names = list(analyzers.keys())
        
        # left: mutual information distributions
        ax = axes[0]
        data_mi = []
        
        for prior_name in prior_names:
            mi_stats = analyzers[prior_name].analyze_mutual_information()
            scores = mi_stats.get("scores", np.array([]))
            data_mi.append(scores)
                
        positions = np.arange(len(prior_names))
        vp = ax.violinplot(data_mi, positions=positions, widths=0.7,
                          showmeans=True, showmedians=True)
        
        for i, pc in enumerate(vp['bodies']):
            pc.set_facecolor(colors[prior_names[i]])
            pc.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel('Prior')
        ax.set_ylabel('Mutual Information')
        ax.set_title('Feature-Target Mutual Information Distribution')
        
        # right: mean mutual information
        ax = axes[1]
        means = [np.mean(mi) for mi in data_mi]
        stds = [np.std(mi) for mi in data_mi]
        
        ax.bar(positions, means, yerr=stds, color=[colors[p] for p in prior_names],
              alpha=0.7, capsize=5)
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel('Prior')
        ax.set_ylabel('Mean Mutual Information')
        ax.set_title('Average MI Scores')
        
        plt.tight_layout()
        return fig, axes


def plot_complexity_noise(analyzers: Dict[str, RegressionDataAnalyzer]) -> Tuple[plt.Figure, np.ndarray]:
    """Plot complexity and noise characteristics.
    
    Shows target variability and estimated noise levels.
    
    Args:
        analyzers: Dict mapping prior names to RegressionDataAnalyzer instances
        
    Returns:
        Tuple of (figure, axes)
    """

    with apply_plot_style():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        colors = get_prior_colors(list(analyzers.keys()))
        
        prior_names = list(analyzers.keys())
        positions = np.arange(len(prior_names))

        # collect stats once per prior using the analyzer
        variances = []
        ranges = []
        cvs = []
        skews = []

        for prior_name in prior_names:
            t_stats = analyzers[prior_name].analyze_target_distribution()
            
            variances.append(t_stats["variance"])
            ranges.append(t_stats["range"])
            skews.append(t_stats["skewness"])

            # cv can be inf if mean == 0 in the analyzer
            cv = t_stats.get("coefficient_of_variation", np.nan)
            if np.isinf(cv):
                cv = np.nan
            cvs.append(cv)

        # top-left: target variance
        ax = axes[0, 0]
        ax.bar(positions, variances,
               color=[colors[p] for p in prior_names],
               alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Variance")
        ax.set_title("Target Variance")

        # top-right: target range
        ax = axes[0, 1]
        ax.bar(positions, ranges,
               color=[colors[p] for p in prior_names],
               alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Range")
        ax.set_title("Target Value Range")

        # bottom-left: coefficient of variation
        ax = axes[1, 0]
        ax.bar(positions, cvs,
               color=[colors[p] for p in prior_names],
               alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Coefficient of Variation")
        ax.set_title("Relative Variability")

        # bottom-right: skewness
        ax = axes[1, 1]
        ax.bar(positions, skews,
               color=[colors[p] for p in prior_names],
               alpha=0.7)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel("Prior")
        ax.set_ylabel("Skewness")
        ax.set_title("Target Distribution Skewness")
        
        plt.tight_layout()
        return fig, axes


def plot_redundancy(analyzers: Dict[str, RegressionDataAnalyzer]) -> Tuple[plt.Figure, np.ndarray]:
    """Plot feature redundancy analysis.
    
    Shows inter-feature correlations and redundancy metrics.
    
    Args:
        analyzers: Dict mapping prior names to RegressionDataAnalyzer instances
        
    Returns:
        Tuple of (figure, axes)
    """
    with apply_plot_style():
        n_priors = len(analyzers)
        fig, axes = plt.subplots(1, n_priors + 1, figsize=(4 * (n_priors + 1), 4))
        colors = get_prior_colors(list(analyzers.keys()))
        
        if n_priors == 1:
            axes = [axes]
        
        # heatmaps for each prior
        for idx, (prior_name, analyzer) in enumerate(analyzers.items()):
            ax = axes[idx]
            features = analyzer.get_all_features()
            
            # compute correlation matrix
            if features.shape[1] > 1:
                corr_matrix = np.corrcoef(features.T)
            else:
                corr_matrix = np.array([[1.0]])
            
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax.set_title(f'{prior_name}\nFeature Correlations')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Feature Index')
            
            # add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # summary plot: mean absolute correlation
        ax = axes[-1]
        prior_names = list(analyzers.keys())
        positions = np.arange(len(prior_names))
        mean_corrs = []

        for prior_name in prior_names:
            red_stats = analyzers[prior_name].analyze_feature_redundancy()
            mean_corrs.append(red_stats.get("mean_abs_correlation", 0.0))
        
        ax.bar(positions, mean_corrs, color=[colors[p] for p in prior_names], alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(prior_names, rotation=45)
        ax.set_xlabel('Prior')
        ax.set_ylabel('Mean |Correlation|')
        ax.set_title('Feature Redundancy')
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        return fig, axes
    

# prior summary vector for similarity comparisons
def prior_summary_vector(
    analyzer: RegressionDataAnalyzer,
) -> Tuple[np.ndarray, List[str]]:
    """Build a fixed-length numeric summary vector for one prior.

    Uses existing analyzer methods and only scalar stats.

    Returns:
        (vector, metric_names) so we can keep consistent ordering.
    """
    # dimensionality / basic stats
    basic = analyzer.get_basic_statistics()
    num_feat_stats = basic["num_features"]
    seq_len_stats = basic["seq_lengths"]
    eval_pos_stats = basic["eval_positions"]

    # grab the other stats dictionaries
    t = analyzer.analyze_target_distribution()
    f = analyzer.analyze_feature_distributions()
    rel = analyzer.analyze_target_feature_relationships()
    mi = analyzer.analyze_mutual_information()
    red = analyzer.analyze_feature_redundancy()
    dev = analyzer.analyze_target_scale_and_deviation()
    noise = analyzer.analyze_noise_characteristics()

    # pick a compact set of informative scalar metrics
    metrics: Dict[str, float] = {
        # how many features per task?
        "dim_num_features_mean": float(num_feat_stats.get("mean", 0.0)),
        "dim_num_features_std": float(num_feat_stats.get("std", 0.0)),
        # how many datapoints per task?
        "dim_seq_len_mean": float(seq_len_stats.get("mean", 0.0)),
        "dim_seq_len_std": float(seq_len_stats.get("std", 0.0)),
        # where in the sequence are we evaluated on average?
        "dim_eval_pos_mean": float(eval_pos_stats.get("mean", 0.0)),
        # target distribution (how y behaves globally)
        "t_mean": float(t.get("mean", 0.0)),
        "t_std": float(t.get("std", 0.0)),
        "t_range": float(t.get("range", 0.0)),
        "t_skew": float(t.get("skewness", 0.0)),
        # feature distribution (how x behaves globally)
        "f_mean": float(f.get("mean", 0.0)),
        "f_std": float(f.get("std", 0.0)),
        "f_zero_ratio": float(f.get("zero_ratio", 0.0)),
        # feature–target relationships
        "rel_pearson_mean": float(rel.get("pearson_mean_abs", 0.0)),
        "rel_spearman_mean": float(rel.get("spearman_mean_abs", 0.0)),
        "rel_nonlin": float(rel.get("nonlinearity_score", 0.0)),
        "rel_inf_ratio": float(rel.get("informative_features_ratio", 0.0)),
        # mutual information (non-linear dependence, summarized)
        "mi_mean": float(mi.get("mean", 0.0)),
        "mi_q75": float(mi.get("q75", 0.0)),
        # redundancy between features
        "red_mean_abs_corr": float(red.get("mean_abs_correlation", 0.0)),
        "red_high_corr_ratio": float(red.get("high_correlation_ratio", 0.0)),
        # function scale / deviation across tasks
        "dev_mean_std": float(dev.get("mean_target_deviation", 0.0)),
        "dev_mean_range": float(dev.get("mean_target_range", 0.0)),
        # noise / linear fit difficulty
        "noise_mean_std": float(noise.get("mean_noise_std", 0.0)),
        "noise_mean_r2": float(noise.get("mean_linear_r2", 0.0)),
    }

    # keep a semantic order
    metric_names: List[str] = list(metrics.keys())

    # build the vector
    vec = np.array([metrics[name] for name in metric_names], dtype=float)

    # sanitize so similarity computations don't explode
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    return vec, metric_names


def plot_prior_similarity(analyzers: Dict[str, RegressionDataAnalyzer]) -> Tuple[plt.Figure, plt.Axes]:
    """Compare priors to each other based on analyzer summary statistics.

    Builds a summary vector per prior and plots a correlation heatmap.
    High correlation ⇒ priors behave similarly as data generators.
    """
    with apply_plot_style():
        prior_names = list(analyzers.keys())
        n_priors = len(prior_names)

        # build summary matrix (n_priors × n_metrics)
        summary_vectors = []

        for name in prior_names:
            vec, _ = prior_summary_vector(analyzers[name])
            summary_vectors.append(vec)

        summary_matrix = np.vstack(summary_vectors)

        # standardize metrics for scale invariance
        mean = summary_matrix.mean(axis=0, keepdims=True)
        std = summary_matrix.std(axis=0, keepdims=True)
        std = std + 1e-8
        summary_z = (summary_matrix - mean) / std

        # prior–prior similarity via correlation of these vectors
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