"""Analyzer for classification data generated from synthetic priors."""

import os
from typing import Dict, Any

import numpy as np
from sklearn.feature_selection import mutual_info_classif, f_classif

from tfmplayground.priors.experiments.data_analysis import DataAnalyzer


class ClassificationDataAnalyzer(DataAnalyzer):
    """Analyzes classification datasets generated from synthetic priors.

    Computes summary statistics to compare different prior characteristics:
    - Label distribution and class imbalance
    - Feature distributions
    - Feature–label relationships (ANOVA F)
    - Feature redundancy (correlation structure)
    - Mutual information between features and labels

    Assumes the same HDF5 structure as RegressionDataAnalyzer, but with
    discrete class labels in ``y``.
    """

    def get_all_features(self) -> np.ndarray:
        """Get all features from all samples (excluding padding).
        
        Returns:
            Array of shape (total_points, max_features) with all feature values
        """
        features = []
        max_features = 0
        
        # First pass: determine max features from actual data shapes
        for i in range(len(self.data["X"])):
            n_features_actual = self.data["X"][i].shape[1]
            max_features = max(max_features, n_features_actual)
        
        # Second pass: collect features with padding/truncation if needed
        for i in range(len(self.data["X"])):
            n_points = self.data["num_datapoints"][i]
            n_features_actual = self.data["X"][i].shape[1]
            x = self.data["X"][i, :n_points, :n_features_actual]
            
            # Create padded array and copy data
            x_padded = np.full((n_points, max_features), np.nan)
            n_copy = min(n_features_actual, max_features)
            x_padded[:, :n_copy] = x[:, :n_copy]
            
            features.append(x_padded)
        
        return np.vstack(features)
    
    def get_all_targets(self) -> np.ndarray:
        """Get all class labels from all samples (excluding padding).
        
        Returns:
            Array of shape (total_points,) with all class labels
        """
        labels = []
        for i in range(len(self.data["y"])):
            n_points = self.data["num_datapoints"][i]
            labels.append(self.data["y"][i, :n_points].astype(int))
        return np.concatenate(labels)


    def analyze_target_distribution(self) -> Dict:
        """Analyze the distribution of class labels.

        Returns:
            Dictionary with class distribution statistics.
        """
        # collect all non-padded labels
        labels = self.get_all_targets()
        values, counts = np.unique(labels, return_counts=True)
        total = counts.sum()
        probs = counts / total

        # basic distribution info
        stats_dict: Dict[str, Any] = {
            "num_classes": int(len(values)),
            "n_samples": int(total),
            "classes": values.tolist(),
            "class_counts": counts.tolist(),
            "class_probs": probs.tolist(),
        }

        # imbalance metrics
        max_count = counts.max()
        min_count = counts.min()
        stats_dict["majority_class"] = int(values[np.argmax(counts)])
        stats_dict["minority_class"] = int(values[np.argmin(counts)])
        stats_dict["majority_ratio"] = float(max_count / total)
        stats_dict["minority_ratio"] = float(min_count / total)
        stats_dict["imbalance_ratio"] = float(max_count / max(1, min_count))

        # entropy of label distribution
        entropy = -(probs * np.log2(probs + 1e-12)).sum()
        stats_dict["entropy_bits"] = float(entropy)

        return stats_dict


    def analyze_target_feature_relationships(self, n_samples: int = 100) -> Dict:
        """Analyze relationships between features and class labels.

        Uses ANOVA F-statistics aggregated across tasks as a proxy for how
        discriminative features are for the classes.
        """
        f_scores = []

        sample_indices = np.random.choice(
            len(self.data["X"]),
            min(n_samples, len(self.data["X"])),
            replace=False,
        )

        for i in sample_indices:
            n_points = int(self.data["num_datapoints"][i])
            n_features = int(self.data["num_features"][i])
            if n_points < 5 or n_features < 1:
                continue

            X_sample = self.data["X"][i, :n_points, :n_features]
            y_sample = self.data["y"][i, :n_points].astype(int)

            # need at least 2 classes to compute F-statistics
            if len(np.unique(y_sample)) < 2:
                continue

            try:
                f_vals, p_vals = f_classif(X_sample, y_sample)
                # aggregate F scores only; p-values are often extreme for big n
                f_scores.extend(f_vals)
            except Exception:
                continue

        rel_stats: Dict[str, Any] = {}
        if f_scores:
            f_scores_arr = np.array(f_scores)
            rel_stats = {
                "f_mean": float(f_scores_arr.mean()),
                "f_std": float(f_scores_arr.std()),
                "f_max": float(f_scores_arr.max()),
                "f_median": float(np.median(f_scores_arr)),
                "f_q25": float(np.percentile(f_scores_arr, 25)),
                "f_q75": float(np.percentile(f_scores_arr, 75)),
            }

        return rel_stats

    def analyze_mutual_information(self, n_samples: int = 100) -> Dict:
        """Analyze mutual information between features and class labels.

        Uses sklearn's ``mutual_info_classif`` to capture nonlinear
        dependencies between continuous features and discrete labels.
        """
        mi_scores = []

        sample_indices = np.random.choice(
            len(self.data["X"]),
            min(n_samples, len(self.data["X"])),
            replace=False,
        )

        for i in sample_indices:
            n_points = int(self.data["num_datapoints"][i])
            n_features = int(self.data["num_features"][i])
            if n_points < 10 or n_features < 1:
                continue

            X_sample = self.data["X"][i, :n_points, :n_features]
            y_sample = self.data["y"][i, :n_points].astype(int)

            if len(np.unique(y_sample)) < 2:
                continue

            try:
                mi = mutual_info_classif(X_sample, y_sample, random_state=42)
                mi_scores.extend(mi)
            except Exception:
                continue

        if not mi_scores:
            return {}

        mi_arr = np.array(mi_scores)
        mi_stats: Dict[str, Any] = {
            "mean": float(mi_arr.mean()),
            "std": float(mi_arr.std()),
            "max": float(mi_arr.max()),
            "min": float(mi_arr.min()),
            "median": float(np.median(mi_arr)),
            "q75": float(np.percentile(mi_arr, 75)),
        }

        return mi_stats


    def generate_report(self) -> str:
        """Generate a comprehensive text report for classification data.

        Returns:
            Formatted string report.
        """

        report_lines = []
        report_lines.append("=" * 50)
        report_lines.append("CLASSIFICATION DATA ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"File: {os.path.basename(self.h5_path)}")
        report_lines.append("")

        # basic statistics
        report_lines.append("BASIC STATISTICS")
        report_lines.append("-" * 50)
        basic_stats = self.get_basic_statistics()
        report_lines.append(f"Total samples: {basic_stats['total_samples']}")
        report_lines.append(f"Max sequence length: {basic_stats['max_seq_len']}")
        report_lines.append(f"Max features: {basic_stats['max_features']}")
        report_lines.append("")

        report_lines.append("Actual sequence lengths:")
        for key, val in basic_stats["seq_lengths"].items():
            report_lines.append(f"  {key}: {val:.2f}")
        report_lines.append("")

        report_lines.append("Actual number of features:")
        for key, val in basic_stats["num_features"].items():
            report_lines.append(f"  {key}: {val:.2f}")
        report_lines.append("")

        report_lines.append("Evaluation positions:")
        for key, val in basic_stats["eval_positions"].items():
            report_lines.append(f"  {key}: {val:.2f}")
        report_lines.append("")

        # label distribution
        report_lines.append("LABEL DISTRIBUTION")
        report_lines.append("-" * 50)
        label_stats = self.analyze_target_distribution()
        for key, val in label_stats.items():
            report_lines.append(f"{key}: {val}")
        report_lines.append("")

        # feature distribution
        report_lines.append("FEATURE DISTRIBUTION")
        report_lines.append("-" * 50)
        feature_stats = self.analyze_feature_distributions()
        for key, val in feature_stats.items():
            if isinstance(val, (int, np.integer)):
                report_lines.append(f"{key}: {val}")
            elif np.isinf(val):
                report_lines.append(f"{key}: inf")
            else:
                report_lines.append(f"{key}: {val:.4f}")
        report_lines.append("")

        # redundancy
        report_lines.append("FEATURE REDUNDANCY (Collinearity)")
        report_lines.append("-" * 50)
        redundancy_stats = self.analyze_feature_redundancy()
        if redundancy_stats:
            for key, val in redundancy_stats.items():
                report_lines.append(f"{key}: {val:.4f}")
        else:
            report_lines.append("No redundancy data available")
        report_lines.append("")

        # feature-label relationships
        report_lines.append("FEATURE-LABEL RELATIONSHIPS (ANOVA F)")
        report_lines.append("-" * 50)
        rel_stats = self.analyze_target_feature_relationships()
        if rel_stats:
            for key, val in rel_stats.items():
                report_lines.append(f"{key}: {val:.4f}")
        else:
            report_lines.append("No relationship data available")
        report_lines.append("")

        # mutual information
        report_lines.append("MUTUAL INFORMATION (Nonlinear Dependencies)")
        report_lines.append("-" * 50)
        mi_stats = self.analyze_mutual_information()
        if mi_stats:
            for key, val in mi_stats.items():
                report_lines.append(f"{key}: {val:.4f}")
        else:
            report_lines.append("No mutual information data available")
        report_lines.append("")

        report_lines.append("=" * 50)
        return "\n".join(report_lines)


def compare_classification_priors(
    analyzer1: ClassificationDataAnalyzer,
    analyzer2: ClassificationDataAnalyzer,
    name1: str,
    name2: str,
) -> str:
    """Compare two different classification priors side by side.

    The structure mirrors ``compare_regression_priors`` but uses
    classification-specific statistics.
    """
    report_lines: list[str] = []
    report_lines.append("=" * 80)
    report_lines.append(f"CLASSIFICATION COMPARISON: {name1} vs {name2}")
    report_lines.append("=" * 80)
    report_lines.append("")

    # label distributions
    target1 = analyzer1.analyze_target_distribution()
    target2 = analyzer2.analyze_target_distribution()

    report_lines.append("LABEL DISTRIBUTION COMPARISON")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Metric':<30} {name1:<20} {name2:<20} {'Diff':<15}")
    report_lines.append("-" * 80)
    for key in target1.keys():
        v1 = target1[key]
        v2 = target2.get(key, None)
        if v2 is None:
            continue

        if isinstance(v1, (list, dict)):
            # for structured entries we just print both; diff is not meaningful
            report_lines.append(f"{key:<30} {str(v1):<20} {str(v2):<20} {'-':<15}")
        elif isinstance(v1, (int, np.integer)):
            diff = v2 - v1
            report_lines.append(f"{key:<30} {v1:<20} {v2:<20} {diff:<15}")
        else:
            diff = float(v2) - float(v1)
            report_lines.append(f"{key:<30} {float(v1):<20.4f} {float(v2):<20.4f} {diff:<15.4f}")
    report_lines.append("")

    # feature distributions
    feature1 = analyzer1.analyze_feature_distributions()
    feature2 = analyzer2.analyze_feature_distributions()

    report_lines.append("FEATURE DISTRIBUTION COMPARISON")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Metric':<30} {name1:<20} {name2:<20} {'Diff':<15}")
    report_lines.append("-" * 80)
    for key in feature1.keys():
        v1 = feature1[key]
        v2 = feature2.get(key, None)
        if v2 is None:
            continue

        if np.isinf(v1) or np.isinf(v2):
            report_lines.append(f"{key:<30} {'inf':<20} {'inf':<20} {'-':<15}")
        else:
            diff = float(v2) - float(v1)
            report_lines.append(f"{key:<30} {float(v1):<20.4f} {float(v2):<20.4f} {diff:<15.4f}")
    report_lines.append("")

    # feature-label relationships (F)
    rel1 = analyzer1.analyze_target_feature_relationships()
    rel2 = analyzer2.analyze_target_feature_relationships()

    if rel1 and rel2:
        report_lines.append("FEATURE-LABEL RELATIONSHIPS COMPARISON (ANOVA F)")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Metric':<30} {name1:<20} {name2:<20} {'Diff':<15}")
        report_lines.append("-" * 80)
        for key in rel1.keys():
            if key not in rel2:
                continue
            diff = float(rel2[key]) - float(rel1[key])
            report_lines.append(
                f"{key:<30} {float(rel1[key]):<20.4f} {float(rel2[key]):<20.4f} {diff:<15.4f}"
            )
        report_lines.append("")

    # mutual information
    mi1 = analyzer1.analyze_mutual_information()
    mi2 = analyzer2.analyze_mutual_information()

    if mi1 and mi2:
        report_lines.append("MUTUAL INFORMATION COMPARISON")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Metric':<30} {name1:<20} {name2:<20} {'Diff':<15}")
        report_lines.append("-" * 80)
        for key in mi1.keys():
            if key not in mi2:
                continue
            diff = float(mi2[key]) - float(mi1[key])
            report_lines.append(
                f"{key:<30} {float(mi1[key]):<20.4f} {float(mi2[key]):<20.4f} {diff:<15.4f}"
            )
        report_lines.append("")

    # redundancy
    redundancy1 = analyzer1.analyze_feature_redundancy()
    redundancy2 = analyzer2.analyze_feature_redundancy()

    if redundancy1 and redundancy2:
        report_lines.append("FEATURE REDUNDANCY COMPARISON")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Metric':<30} {name1:<20} {name2:<20} {'Diff':<15}")
        report_lines.append("-" * 80)
        for key in redundancy1.keys():
            if key not in redundancy2:
                continue
            diff = float(redundancy2[key]) - float(redundancy1[key])
            report_lines.append(
                f"{key:<30} {float(redundancy1[key]):<20.4f} {float(redundancy2[key]):<20.4f} {diff:<15.4f}"
            )
        report_lines.append("")

    # summary
    report_lines.append("KEY DIFFERENCES SUMMARY")
    report_lines.append("-" * 80)

    # label imbalance
    if "imbalance_ratio" in target1 and "imbalance_ratio" in target2:
        ir_diff = float(target2["imbalance_ratio"]) - float(target1["imbalance_ratio"])
        if abs(ir_diff) > 0.1:
            report_lines.append(
                f"• Class imbalance: {name2} is {'more' if ir_diff > 0 else 'less'} imbalanced "
                f"(Δ imbalance_ratio = {abs(ir_diff):.4f})"
            )

    # label entropy
    if "entropy_bits" in target1 and "entropy_bits" in target2:
        ent_diff = float(target2["entropy_bits"]) - float(target1["entropy_bits"])
        if abs(ent_diff) > 0.05:
            report_lines.append(
                f"• Label entropy: {name2} has {'higher' if ent_diff > 0 else 'lower'} "
                f"class-distribution entropy (Δ = {abs(ent_diff):.4f} bits)"
            )

    # discriminative power (F mean)
    if "f_mean" in rel1 and "f_mean" in rel2:
        fmean_diff = float(rel2["f_mean"]) - float(rel1["f_mean"])
        if abs(fmean_diff) > 0.05:
            report_lines.append(
                f"• Discriminative features (F-mean): {name2} has "
                f"{'more' if fmean_diff > 0 else 'less'} discriminative features "
                f"on average (Δ = {abs(fmean_diff):.4f})"
            )

    # mutual information
    if "mean" in mi1 and "mean" in mi2:
        mimean_diff = float(mi2["mean"]) - float(mi1["mean"])
        if abs(mimean_diff) > 0.01:
            report_lines.append(
                f"• Mutual information: {name2} has "
                f"{'higher' if mimean_diff > 0 else 'lower'} average MI between features and labels "
                f"(Δ = {abs(mimean_diff):.4f})"
            )

    report_lines.append("")
    report_lines.append("=" * 80)
    return "\n".join(report_lines)