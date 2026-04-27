"""Analyzer for classification data generated from synthetic priors."""

import os
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats as scipy_stats
from sklearn import config_context
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from tfmplayground.priors.experiments.base_analyzer import DataAnalyzer


class ClassificationDataAnalyzer(DataAnalyzer):
    """Analyzes classification datasets generated from synthetic priors.

    Computes summary statistics to compare different prior characteristics:
    - Label distribution and class imbalance
    - Feature distributions
    - Feature–label relationships (ANOVA F)
    - Feature redundancy (correlation structure)
    - Mutual information between features and labels
    - Linear model training fit (logistic regression balanced accuracy)

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
        
        # First pass: determine max features from recorded per-sample widths.
        # Do not use X.shape[1], which is often a padded global width.
        for i in range(len(self.data["X"])):
            n_features_actual = int(self.data["num_features"][i])
            max_features = max(max_features, n_features_actual)
        
        # Second pass: collect features with padding/truncation if needed
        for i in range(len(self.data["X"])):
            n_points = self.data["num_datapoints"][i]
            n_features_actual = int(self.data["num_features"][i])
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

    @staticmethod
    def _filter_valid_labels(labels: np.ndarray) -> np.ndarray:
        """Drop masked or invalid class labels before plotting/statistics."""
        labels = np.asarray(labels)
        return labels[np.isfinite(labels) & (labels >= 0)].astype(int, copy=False)


    def analyze_target_distribution(self) -> Dict:
        """Analyze the distribution of class labels.

        Returns:
            Dictionary with per-task-averaged class distribution statistics.
        """
        per_task_num_classes: List[float] = []
        per_task_majority_ratio: List[float] = []
        per_task_minority_ratio: List[float] = []
        per_task_imbalance_ratio: List[float] = []
        per_task_entropy: List[float] = []
        total_points = 0

        for i in range(len(self.data["y"])):
            n_points = int(self.data["num_datapoints"][i])
            raw = self.data["y"][i, :n_points]
            labels = self._filter_valid_labels(raw)
            if labels.size == 0:
                continue

            values, counts = np.unique(labels, return_counts=True)
            total = counts.sum()
            total_points += total
            probs = counts / total

            per_task_num_classes.append(float(len(values)))
            per_task_majority_ratio.append(float(counts.max() / total))
            per_task_minority_ratio.append(float(counts.min() / total))
            per_task_imbalance_ratio.append(float(counts.max() / max(1, counts.min())))
            entropy = -(probs * np.log2(probs + 1e-12)).sum()
            per_task_entropy.append(float(entropy))

        n_tasks = len(per_task_num_classes)
        if n_tasks == 0:
            return {
                "num_classes": 0,
                "n_tasks": 0,
                "n_samples_total": 0,
                "majority_ratio": 0.0,
                "minority_ratio": 0.0,
                "imbalance_ratio": 0.0,
                "entropy_bits": 0.0,
            }

        return {
            "num_classes": float(np.mean(per_task_num_classes)),
            "num_classes_std": float(np.std(per_task_num_classes)),
            "n_tasks": n_tasks,
            "n_samples_total": int(total_points),
            "majority_ratio": float(np.mean(per_task_majority_ratio)),
            "minority_ratio": float(np.mean(per_task_minority_ratio)),
            "imbalance_ratio": float(np.mean(per_task_imbalance_ratio)),
            "entropy_bits": float(np.mean(per_task_entropy)),
        }


    def analyze_target_feature_relationships(self) -> Dict:
        """Analyze relationships between features and class labels.

        Uses ANOVA F-statistics aggregated across tasks as a proxy for how
        discriminative features are for the classes.
        """
        f_scores = []

        for i in range(len(self.data["X"])):
            n_points = int(self.data["num_datapoints"][i])
            n_features = int(self.data["num_features"][i])
            if n_points < 5 or n_features < 1:
                continue

            X_sample = self.data["X"][i, :n_points, :n_features]
            y_raw = self.data["y"][i, :n_points]

            # filter rows with invalid y or X values before F-test
            valid_rows = (
                np.isfinite(y_raw)
                & (y_raw >= 0)
                & np.all(np.isfinite(X_sample), axis=1)
            )
            if valid_rows.sum() < 5:
                continue

            X_sample = X_sample[valid_rows]
            y_sample = y_raw[valid_rows].astype(int)

            # need at least 2 classes to compute F-statistics
            if len(np.unique(y_sample)) < 2:
                continue

            try:
                # drop constant (zero-variance) features
                var = np.nanvar(X_sample, axis=0)
                non_constant = var > 0

                if non_constant.sum() < 1:
                    continue

                X_nc = X_sample[:, non_constant]
                
                with warnings.catch_warnings():
                    # synthetic priors (like tabforest) often generate zero-variance sets 
                    # for identical classes, pushing MSW=0. Scikit-learn will warn and return inf.
                    warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero")
                    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value")

                    f_vals, p_vals = f_classif(X_nc, y_sample)
                
                # cap the infinite F-scores to a numerical bound
                f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=1e6, neginf=0.0)
                f_vals = np.clip(f_vals, 0.0, None)
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

    def analyze_mutual_information(self) -> Dict:
        """Analyze mutual information between features and class labels.

        Uses sklearn's ``mutual_info_classif`` to capture nonlinear
        dependencies between continuous features and discrete labels.
        """
        mi_scores = []
        n_samples = int(self.analysis_config["n_samples_mi"])
        if n_samples <= 0:
            return {}

        rng = np.random.default_rng(self.random_state)
        sample_indices = rng.choice(
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
            y_raw = self.data["y"][i, :n_points]
            valid_rows = np.isfinite(y_raw) & (y_raw >= 0) & np.all(np.isfinite(X_sample), axis=1)
            if valid_rows.sum() < 10:
                continue

            X_sample = X_sample[valid_rows]
            y_sample = y_raw[valid_rows].astype(int)

            if len(np.unique(y_sample)) < 2:
                continue

            try:
                non_constant = np.nanvar(X_sample, axis=0) > 1e-12
                if non_constant.sum() < 1:
                    continue

                X_mi = np.ascontiguousarray(
                    X_sample[:, non_constant],
                    dtype=np.float64,
                )
                with config_context(enable_cython_pairwise_dist=False):
                    mi = mutual_info_classif(
                        X_mi,
                        y_sample,
                        random_state=self.random_state,
                    )
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


    def prior_summary_vector(self) -> Tuple[np.ndarray, List[str]]:
        """Build a fixed-length numeric summary vector for this classification prior."""
        target_stats = self.analyze_target_distribution()
        rel_stats = self.analyze_target_feature_relationships()
        mi_stats = self.analyze_mutual_information()
        red_stats = self.analyze_feature_redundancy()
        derived = _compute_classification_similarity_features(self)

        # log-transform F-statistics to compress scale and avoid outliers
        f_mean = max(0.0, float(rel_stats.get("f_mean", 0.0)))
        f_median = max(0.0, float(rel_stats.get("f_median", 0.0)))
        log_f_mean = float(np.log1p(f_mean))
        log_f_median = float(np.log1p(f_median))

        metrics: Dict[str, float] = {
            "class_num_classes": float(target_stats.get("num_classes", 0.0)),
            "class_majority_ratio": float(target_stats.get("majority_ratio", 0.0)),
            "class_entropy": float(target_stats.get("entropy_bits", 0.0)),
            "sep_log_f_mean": log_f_mean,
            "sep_log_f_median": log_f_median,
            "mi_mean": float(mi_stats.get("mean", 0.0)),
            "mi_q75": float(mi_stats.get("q75", 0.0)),
            "class_sep_smd_median": float(derived.get("class_sep_smd_median", 0.0)),
            "red_mean_abs_corr": float(red_stats.get("mean_abs_correlation", 0.0)),
            "pca_top1_var": float(derived.get("pca_top1_var", 0.0)),
            "effective_rank_ratio": float(derived.get("effective_rank_ratio", 0.0)),
            "nonlinear_gap": float(derived.get("nonlinear_feature_target_gap", 0.0)),
            # Feature distribution shape
            "feat_kurtosis_median": float(derived.get("feat_kurtosis_median", 0.0)),
            "feat_discrete_ratio": float(derived.get("feat_discrete_ratio", 0.0)),
            "linear_train_bal_acc": float(derived.get("linear_train_bal_acc", 0.0)),
        }

        metric_names: List[str] = list(metrics.keys())
        vec = np.array([metrics[name] for name in metric_names], dtype=float)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        return vec, metric_names

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
            line = self._format_stat_line(key, val)
            if line is not None:
                report_lines.append(line)
        report_lines.append("")

        # feature distribution
        report_lines.append("FEATURE DISTRIBUTION")
        report_lines.append("-" * 50)
        feature_stats = self.analyze_feature_distributions()
        for key, val in feature_stats.items():
            line = self._format_stat_line(key, val)
            if line is not None:
                report_lines.append(line)
        report_lines.append("")

        # redundancy
        report_lines.append("FEATURE REDUNDANCY (Collinearity)")
        report_lines.append("-" * 50)
        redundancy_stats = self.analyze_feature_redundancy()
        if redundancy_stats:
            for key, val in redundancy_stats.items():
                line = self._format_stat_line(key, val)
                if line is not None:
                    report_lines.append(line)
        else:
            report_lines.append("No redundancy data available")
        report_lines.append("")

        # feature-label relationships
        report_lines.append("FEATURE-LABEL RELATIONSHIPS (ANOVA F)")
        report_lines.append("-" * 50)
        rel_stats = self.analyze_target_feature_relationships()
        if rel_stats:
            for key, val in rel_stats.items():
                line = self._format_stat_line(key, val)
                if line is not None:
                    report_lines.append(line)
        else:
            report_lines.append("No relationship data available")
        report_lines.append("")

        # mutual information
        report_lines.append("MUTUAL INFORMATION (Nonlinear Dependencies)")
        report_lines.append("-" * 50)
        mi_stats = self.analyze_mutual_information()
        if mi_stats:
            for key, val in mi_stats.items():
                line = self._format_stat_line(key, val)
                if line is not None:
                    report_lines.append(line)
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


# prior similarity computation
def _pairwise_class_smd_values(X: np.ndarray, y: np.ndarray) -> List[float]:
    """Compute absolute pairwise class SMD values across all features."""
    classes = np.unique(y)
    if classes.size < 2:
        return []

    smd_values: List[float] = []
    for idx_a in range(len(classes)):
        for idx_b in range(idx_a + 1, len(classes)):
            a = X[y == classes[idx_a]]
            b = X[y == classes[idx_b]]
            if a.shape[0] < 2 or b.shape[0] < 2:
                continue
            mean_diff = np.abs(np.nanmean(a, axis=0) - np.nanmean(b, axis=0))
            var_a = np.nanvar(a, axis=0)
            var_b = np.nanvar(b, axis=0)
            pooled_std = np.sqrt(0.5 * (var_a + var_b)) + 1e-12
            smd = np.nan_to_num(mean_diff / pooled_std, nan=0.0, posinf=0.0, neginf=0.0)
            smd_values.extend(smd.tolist())

    return smd_values


def _compute_classification_similarity_features(
    analyzer: ClassificationDataAnalyzer,
    mi_signal_threshold: float = 1e-3,
) -> Dict[str, float]:
    """Compute data-only derived metrics for prior similarity.

    Args:
        analyzer: A loaded classification analyzer.
        mi_signal_threshold: MI cutoff used to decide whether a feature
            carries signal.
    """
    sample_count = len(analyzer.data["X"])
    if sample_count == 0:
        return {}

    mi_top1_ratios: List[float] = []
    mi_top5_ratios: List[float] = []
    frac_signal_features_values: List[float] = []
    class_sep_smd_mean_values: List[float] = []
    class_sep_smd_median_values: List[float] = []
    class_sep_smd_q75_values: List[float] = []
    pca_top1_values: List[float] = []
    pca_top5_values: List[float] = []
    effective_rank_values: List[float] = []
    effective_rank_ratio_values: List[float] = []
    nonlinear_gap_values: List[float] = []
    kurtosis_values: List[float] = []
    discrete_ratio_values: List[float] = []
    lin_sep_acc_values: List[float] = []
    lin_task_count = 0
    max_lin_tasks = 100  # cap for the LogisticRegression cost


    for i in range(sample_count):
        n_points = int(analyzer.data["num_datapoints"][i])
        n_features = int(analyzer.data["num_features"][i])
        if n_points < 10 or n_features < 1:
            continue

        X = analyzer.data["X"][i, :n_points, :n_features]
        y_raw = analyzer.data["y"][i, :n_points]

        valid_mask = (
            np.isfinite(y_raw)
            & (y_raw >= 0)
            & np.all(np.isfinite(X), axis=1)
        )
        if valid_mask.sum() < 10:
            continue

        y = y_raw[valid_mask].astype(int)
        X = X[valid_mask]
        if np.unique(y).size < 2:
            continue

        col_std = np.std(X, axis=0)
        non_constant = col_std > 1e-12
        if non_constant.sum() < 1:
            continue
        X_nc = X[:, non_constant]

        # feature distribution shape
        try:
            kurt_per_feat = scipy_stats.kurtosis(X_nc, axis=0)
            kurtosis_values.append(float(np.median(np.nan_to_num(kurt_per_feat, nan=0.0))))
        except Exception:
            pass
        n_unique = np.array([len(np.unique(X_nc[:, j])) for j in range(X_nc.shape[1])])
        discrete_ratio_values.append(float(np.mean(n_unique <= 10)))

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                f_vals, _ = f_classif(X_nc, y)

            # after row filtering, some columns may have become constant; drop those before MI
            non_constant_mi = np.var(X_nc, axis=0) > 1e-12
            if non_constant_mi.sum() < 1:
                continue
            X_mi = np.ascontiguousarray(
                X_nc[:, non_constant_mi],
                dtype=np.float64,
            )
            with config_context(enable_cython_pairwise_dist=False):
                mi = mutual_info_classif(
                    X_mi,
                    y,
                    random_state=analyzer.random_state,
                )
            f_vals = f_vals[non_constant_mi]
        except Exception:
            continue

        f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=1e6, neginf=0.0)
        mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)

        mi_sorted = np.sort(mi)[::-1]
        mi_sum = float(mi_sorted.sum())
        if mi_sum > 0.0:
            mi_top1_ratios.append(float(mi_sorted[0] / mi_sum))
            mi_top5_ratios.append(float(mi_sorted[:5].sum() / mi_sum))
        else:
            mi_top1_ratios.append(0.0)
            mi_top5_ratios.append(0.0)
        frac_signal_features_values.append(float(np.mean(mi > mi_signal_threshold)))

        if mi.size > 0 and f_vals.size > 0:
            mi_cut = np.percentile(mi, 75)
            f_cut = np.percentile(f_vals, 25)
            nonlinear_gap_values.append(float(np.mean((mi >= mi_cut) & (f_vals <= f_cut))))

        task_smd = _pairwise_class_smd_values(X_nc, y)
        if task_smd:
            task_smd_arr = np.array(task_smd, dtype=float)
            class_sep_smd_mean_values.append(float(np.mean(task_smd_arr)))
            class_sep_smd_median_values.append(float(np.median(task_smd_arr)))
            class_sep_smd_q75_values.append(float(np.percentile(task_smd_arr, 75)))

        # In-sample logistic regression balanced accuracy.
        if lin_task_count < max_lin_tasks:
            try:
                lr = LogisticRegression(
                    max_iter=300, random_state=analyzer.random_state,
                    C=1.0, solver="lbfgs",
                )
                lr.fit(X_nc, y)
                y_pred = lr.predict(X_nc)
                lin_sep_acc_values.append(
                    float(balanced_accuracy_score(y, y_pred))
                )
                lin_task_count += 1
            except Exception:
                pass

        X_centered = X_nc - np.nanmean(X_nc, axis=0, keepdims=True)
        try:
            cov = np.cov(X_centered, rowvar=False)
            eigvals = np.linalg.eigvalsh(cov)
        except Exception:
            continue

        eigvals = np.clip(np.nan_to_num(eigvals, nan=0.0, posinf=0.0, neginf=0.0), a_min=0.0, a_max=None)
        total_var = float(eigvals.sum())
        if total_var <= 0.0:
            continue

        eigvals_desc = eigvals[::-1]
        pca_top1_values.append(float(eigvals_desc[0] / total_var))
        pca_top5_values.append(float(eigvals_desc[:5].sum() / total_var))

        probs = eigvals_desc / total_var
        probs = probs[probs > 1e-12]
        eff_rank = float(np.exp(-np.sum(probs * np.log(probs))))
        effective_rank_values.append(eff_rank)
        n_nonconstant = int(X_nc.shape[1])
        effective_rank_ratio_values.append(
            eff_rank / n_nonconstant if n_nonconstant > 0 else 0.0
        )

    result: Dict[str, float] = {}

    if mi_top1_ratios:
        result["mi_top1_ratio"] = float(np.mean(mi_top1_ratios))
    if mi_top5_ratios:
        result["mi_top5_ratio"] = float(np.mean(mi_top5_ratios))
    if frac_signal_features_values:
        result["frac_signal_features"] = float(np.mean(frac_signal_features_values))

    if class_sep_smd_mean_values:
        result["class_sep_smd_mean"] = float(np.mean(class_sep_smd_mean_values))
    if class_sep_smd_median_values:
        result["class_sep_smd_median"] = float(np.mean(class_sep_smd_median_values))
    if class_sep_smd_q75_values:
        result["class_sep_smd_q75"] = float(np.mean(class_sep_smd_q75_values))

    if pca_top1_values:
        result["pca_top1_var"] = float(np.mean(pca_top1_values))
    if pca_top5_values:
        result["pca_top5_var"] = float(np.mean(pca_top5_values))
    if effective_rank_values:
        result["effective_rank"] = float(np.mean(effective_rank_values))
    if effective_rank_ratio_values:
        result["effective_rank_ratio"] = float(np.mean(effective_rank_ratio_values))
    if nonlinear_gap_values:
        result["nonlinear_feature_target_gap"] = float(np.mean(nonlinear_gap_values))

    if kurtosis_values:
        result["feat_kurtosis_median"] = float(np.median(kurtosis_values))
    if discrete_ratio_values:
        result["feat_discrete_ratio"] = float(np.mean(discrete_ratio_values))
    if lin_sep_acc_values:
        result["linear_train_bal_acc"] = float(np.mean(lin_sep_acc_values))

    return result
