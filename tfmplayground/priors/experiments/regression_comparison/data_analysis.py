"""Analyzer for regression data generated from synthetic priors."""

import os
from typing import Dict

import h5py
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression


class RegressionDataAnalyzer:
    """Analyzes regression datasets generated from synthetic priors (GP, MLP, etc.).
    
    Computes comprehensive statistics to compare different prior characteristics:
    - Target and feature distributions
    - Feature-target relationships (linear, nonlinear)
    - Function complexity and noise characteristics
    - Feature redundancy and mutual information
    """

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.data = None
        self.metadata = {}
        self._load_data()
    
    # TODO: Carry this under a common '.py' for experiments cuz it's likely reused in the classification analysis
    def _load_data(self):
        print(f"Loading data from {self.h5_path}...")
        
        with h5py.File(self.h5_path, "r") as f:
            self.data = {
                "X": f["X"][:],
                "y": f["y"][:],
                "num_features": f["num_features"][:],
                "num_datapoints": f["num_datapoints"][:],
                "single_eval_pos": f["single_eval_pos"][:],
            }
            
            self.metadata = {
                "problem_type": f["problem_type"][()].decode("utf-8")
            }
            
        print(f"  Loaded {len(self.data['X'])} samples")
        print(f"  Problem type: {self.metadata['problem_type']}")
        
    def get_basic_statistics(self) -> Dict:
        """Extract basic statistics of the dataset.
        
        Returns:
            Dictionary containing various statistics
        """
        stats_dict = {
            "total_samples": len(self.data["X"]),
            # samples could potentially have different lengths/features
            "max_seq_len": self.data["X"].shape[1],
            "max_features": self.data["X"].shape[2],
        }

        # actual sequence lengths (cuz its padded to maximums)
        stats_dict["seq_lengths"] = {
            "min": int(self.data["num_datapoints"].min()), # shortest sequence length
            "max": int(self.data["num_datapoints"].max()), # longest sequence length
            "mean": float(self.data["num_datapoints"].mean()), # mean sequence length
            "std": float(self.data["num_datapoints"].std()), # std sequence length
        }
        
        # actual number of features used
        stats_dict["num_features"] = {
            "min": int(self.data["num_features"].min()),
            "max": int(self.data["num_features"].max()),
            "mean": float(self.data["num_features"].mean()),
            "std": float(self.data["num_features"].std()),
        }
        
        # analyze evaluation positions
        stats_dict["eval_positions"] = {
            "min": int(self.data["single_eval_pos"].min()),
            "max": int(self.data["single_eval_pos"].max()),
            "mean": float(self.data["single_eval_pos"].mean()),
            "std": float(self.data["single_eval_pos"].std()),
        }
        
        return stats_dict
    
    def analyze_target_distribution(self) -> Dict:
        """Analyze the distribution of target values.
        
        Returns:
            Dictionary with target distribution statistics
        """

        # get the non-padded version of target values
        y_values = []
        for i in range(len(self.data["y"])):
            num_real_points = self.data["num_datapoints"][i]
            # get the i'th y up to the real number of points
            y_values.extend(self.data["y"][i, :num_real_points].flatten())
        
        y_values = np.array(y_values)
        
        target_stats = {
            "mean": float(y_values.mean()),
            "std": float(y_values.std()),
            "variance": float(y_values.var()),
            "min": float(y_values.min()),
            "max": float(y_values.max()),
            "range": float(y_values.max() - y_values.min()),
            "median": float(np.median(y_values)),
            # quartiles: 
            # - q25 (25th percentile) → 25% of data points are below this value
            "q25": float(np.percentile(y_values, 25)),
            "q75": float(np.percentile(y_values, 75)),
            # interquartile range (includes 50% of data)
            "iqr": float(np.percentile(y_values, 75) - np.percentile(y_values, 25)),
            "skewness": float(stats.skew(y_values)),
            "kurtosis": float(stats.kurtosis(y_values)),
        }
        
        # normality test, returns test statistic and p-value (we only need a subset to detect it being normal)
        _, p_value = stats.normaltest(y_values[:min(5000, len(y_values))])
        # store the probability of being normal
        target_stats["normality_p_value"] = float(p_value)
        target_stats["is_normal"] = p_value > 0.05
        
        # detect the outliers using IQR method
        # set the cut-off thresholds to decide outliers
        lower_bound = target_stats["q25"] - 1.5 * target_stats["iqr"]
        upper_bound = target_stats["q75"] + 1.5 * target_stats["iqr"]

        # boolean mask for outliers
        outliers = (y_values < lower_bound) | (y_values > upper_bound)
        target_stats["outlier_ratio"] = float(outliers.sum() / len(y_values))
        target_stats["num_outliers"] = int(outliers.sum())
        
        # coefficient of variation (relative variability to the mean)
        if target_stats["mean"] != 0: # division by zero check
            target_stats["coefficient_of_variation"] = target_stats["std"] / abs(target_stats["mean"])
        else:
            target_stats["coefficient_of_variation"] = float('inf')
        
        return target_stats
    
    def analyze_feature_distributions(self, sample_size: int = 1000) -> Dict:
        """Analyze the distribution of feature values.
        
        Args:
            sample_size: Number of samples to use for analysis
            
        Returns:
            Dictionary with feature distribution statistics
        """

        # randomly choose sample_size amount of indices to keep it memory efficient
        sample_indices = np.random.choice(
            len(self.data["X"]), 
            min(sample_size, len(self.data["X"])), 
            replace=False
        )
        
        # obtain the feature values for those indices & get rid of the paddings
        features = []
        for i in sample_indices:
            n_points = self.data["num_datapoints"][i]
            n_features = self.data["num_features"][i]
            features.extend(self.data["X"][i, :n_points, :n_features].flatten())

        features = np.array(features)

        feature_stats = {
            "mean": float(features.mean()),
            "std": float(features.std()),
            "variance": float(features.var()),
            "min": float(features.min()),
            "max": float(features.max()),
            "range": float(features.max() - features.min()),
            "median": float(np.median(features)),
            "q25": float(np.percentile(features, 25)),
            "q75": float(np.percentile(features, 75)),
            "iqr": float(np.percentile(features, 75) - np.percentile(features, 25)),
            "skewness": float(stats.skew(features)),
            "kurtosis": float(stats.kurtosis(features)),
        }
        
        # outlier detection
        lower_bound = feature_stats["q25"] - 1.5 * feature_stats["iqr"]
        upper_bound = feature_stats["q75"] + 1.5 * feature_stats["iqr"]
        outliers = (features < lower_bound) | (features > upper_bound)
        feature_stats["outlier_ratio"] = float(outliers.sum() / len(features))

        # coefficient of variation
        if feature_stats["mean"] != 0:
            feature_stats["coefficient_of_variation"] = feature_stats["std"] / abs(feature_stats["mean"])
        else:
            feature_stats["coefficient_of_variation"] = float('inf')
        
        # zero ratio (to detect sparsity)
        feature_stats["zero_ratio"] = float((features == 0).sum() / len(features))
        
        return feature_stats
    

    def analyze_target_feature_relationships(self, n_samples: int = 100) -> Dict:
        """Analyze relationships between features and targets.
        
        Args:
            n_samples: Number of samples to analyze (analyzing all is costly)
            
        Returns:
            Dictionary with relationship statistics
        """

        # sample n_samples to analyze
        sample_indices = np.random.choice(
            len(self.data["X"]), 
            min(n_samples, len(self.data["X"])), 
            replace=False
        )

        # measures linear correlation (Pearson)
        # close to 1 or -1 means strong linear relationship 
        pearson_corrs = []
        # and monotonic correlation (Spearman)
        spearman_corrs = []

        # get rid of the padding
        for i in sample_indices:
            n_points = self.data["num_datapoints"][i]
            n_features = self.data["num_features"][i]
            
            X_sample = self.data["X"][i, :n_points, :n_features]
            y_sample = self.data["y"][i, :n_points]
            
            for j in range(n_features):
                
                # pearson correlation for feature j
                # does y increase/decrease linearly with feature j?
                pearson_corr, _ = stats.pearsonr(X_sample[:, j], y_sample)

                if not np.isnan(pearson_corr):
                    pearson_corrs.append(pearson_corr)
                
                # does y increase/decrease monotonically with feature j?
                spearman_corr, _ = stats.spearmanr(X_sample[:, j], y_sample)

                if not np.isnan(spearman_corr):
                    spearman_corrs.append(spearman_corr)
        
        pearson_corrs = np.array(pearson_corrs)
        spearman_corrs = np.array(spearman_corrs)
        
        abs_pearson = np.abs(pearson_corrs)
        abs_spearman = np.abs(spearman_corrs)
        
        rel_stats = {
            # linear correlation (Pearson)
            "pearson_mean_abs": float(abs_pearson.mean()),  # overall correlation strength
            "pearson_max_abs": float(abs_pearson.max()),     # strongest feature
            "pearson_q10": float(np.percentile(abs_pearson, 10)),  # weak features baseline (correlations of the bottom 10% features)
            "pearson_q90": float(np.percentile(abs_pearson, 90)),  # strong features threshold (top 10%)

            # monotonic correlation (Spearman)
            "spearman_mean_abs": float(abs_spearman.mean()),
            "spearman_max_abs": float(abs_spearman.max()),
            "spearman_q10": float(np.percentile(abs_spearman, 10)),
            "spearman_q90": float(np.percentile(abs_spearman, 90)),
            
            # nonlinearity: how much do rankings differ from linear relationships?
            "nonlinearity_score": float(np.abs(spearman_corrs - pearson_corrs).mean()),

            # using threshold of 0.1 for "informative" correlation
            "informative_features_ratio": float((abs_pearson > 0.1).sum() / len(pearson_corrs)),
            
            # direction consistency: are features mostly positively or negatively correlated?
            "positive_correlation_ratio": float((pearson_corrs > 0).sum() / len(pearson_corrs)),
            
            # feature importance variance: consistency of feature usage across samples
            # higher variance = features are used very differently across samples
            "feature_importance_variance": float(abs_pearson.std()),
        }
        
        return rel_stats
    

    def analyze_mutual_information(self, n_samples: int = 100) -> Dict:
        """Analyze mutual information between features and targets.
        
        Mutual information captures both linear and nonlinear dependencies.
        
        Args:
            n_samples: Number of samples to analyze
            
        Returns:
            Dictionary with mutual information statistics
        """
        mi_scores = []
        
        # sample n_samples to analyze
        sample_indices = np.random.choice(
            len(self.data["X"]), 
            min(n_samples, len(self.data["X"])), 
            replace=False
        )
        
        for i in sample_indices:
            n_points = self.data["num_datapoints"][i]
            n_features = self.data["num_features"][i]
            
            # need sufficient to estimate mutual information
            if n_points < 10:
                continue
            
            # extract the non-padded sampled data
            X_sample = self.data["X"][i, :n_points, :n_features]
            y_sample = self.data["y"][i, :n_points]
            
            # compute the mutual information
            # basically the KL divergence between the joint distribution p(xi, y) and the product of marginals p(xi)p(y).
            # MI is larger if knowing feature xi reduces uncertainty about y more
            try:
                mi = mutual_info_regression(X_sample, y_sample, random_state=42)
                # this returns an array of length n_features
                mi_scores.extend(mi)
            except:
                continue
        
        if len(mi_scores) > 0:
            mi_scores = np.array(mi_scores)
            # compute the global stats
            mi_stats = {
                "mean": float(mi_scores.mean()),
                "std": float(mi_scores.std()),
                "max": float(mi_scores.max()),
                "min": float(mi_scores.min()),
                "median": float(np.median(mi_scores)),
                # q75 will show how strong the upper range of feature informativeness is
                "q75": float(np.percentile(mi_scores, 75)),
                # these will help capture the spread of 'easy' vs 'hard' features between various priors
            }
        else:
            mi_stats = {}
        
        return mi_stats
    
    
    def analyze_target_scale_and_deviation(self, n_samples: int = 100) -> Dict:
        """Analyze the target scale and deviation of the regression functions.
        
        Args:
            n_samples: Number of samples to analyze
            
        Returns:
            Dictionary with deviation metrics
        """
        target_deviations = []
        target_ranges = []
        
        # sample n_samples to analyze
        sample_indices = np.random.choice(
            len(self.data["X"]), 
            min(n_samples, len(self.data["X"])), 
            replace=False
        )
        
        for i in sample_indices:
            n_points = self.data["num_datapoints"][i]
            y_sample = self.data["y"][i, :n_points]
            
            target_deviations.append(np.std(y_sample))
            # peak to peak range aka max - min
            target_ranges.append(np.ptp(y_sample))
        
        deviation_stats = {
            "mean_target_deviation": float(np.mean(target_deviations)),
            "std_target_deviation": float(np.std(target_deviations)),
            "mean_target_range": float(np.mean(target_ranges)),
            "std_target_range": float(np.std(target_ranges)),
        }
        
        return deviation_stats
    
    
    def analyze_feature_redundancy(self, n_samples: int = 100) -> Dict:
        """Analyze redundancy and collinearity among features.
        
        Args:
            n_samples: Number of samples to analyze
            
        Returns:
            Dictionary with redundancy metrics
        """

        all_corr = []
        high_corr_pair_counts = []
        
        sample_indices = np.random.choice(
            len(self.data["X"]), 
            min(n_samples, len(self.data["X"])), 
            replace=False
        )
        
        for i in sample_indices:
            n_points = self.data["num_datapoints"][i]
            n_features = self.data["num_features"][i]
            
            # skip the cases with too few features
            if n_features < 2:
                continue
            
            X_sample = self.data["X"][i, :n_points, :n_features]
            
            # compute the correlation matrix
            try:
                # this is of shape n_features x n_features
                corr_matrix = np.corrcoef(X_sample.T)
                
                # get upper triangle (excluding the diagonal aka correlation of feature with itself)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                correlations = corr_matrix[mask]
                # now it has unique feature pair correlations flattened into a 1D array
                
                all_corr.extend(correlations)
                
                # measure the high correlation pairs if corr coefficient is > 0.8
                high_corr = np.abs(correlations) > 0.8
                # collect the number of highly correlated pairs
                high_corr_pair_counts.append(high_corr.sum())
            except:
                continue
        
        if len(all_corr) > 0:
            all_corr = np.array(all_corr)
            redundancy_stats = {
                "mean_abs_correlation": float(np.abs(all_corr).mean()),
                "max_abs_correlation": float(np.abs(all_corr).max()),
                "std_correlation": float(all_corr.std()),
                "high_correlation_ratio": float((np.abs(all_corr) > 0.8).sum() / len(all_corr)),
                "mean_high_corr_pairs": float(np.mean(high_corr_pair_counts)) if high_corr_pair_counts else 0.0,
            }
        else:
            redundancy_stats = {}
        
        return redundancy_stats
    
    
    def analyze_noise_characteristics(self, n_samples: int = 100) -> Dict:
        """Analyze the added noise in the regression functions.
        
        We approximate the noise level by fitting a simple linear model to each dataset
        and treating the remaining residual variance (the part the model cannot explain)
        as an estimate of noise.
                
        Args:
            n_samples: Number of samples to analyze
            
        Returns:
            Dictionary with noise statistics
        """

        noise_estimates = []
        
        sample_indices = np.random.choice(
            len(self.data["X"]), 
            min(n_samples, len(self.data["X"])), 
            replace=False
        )
        
        for i in sample_indices:
            n_points = self.data["num_datapoints"][i]
            n_features = self.data["num_features"][i]
            
            if n_points < 10 or n_features < 1:
                continue
            
            X_sample = self.data["X"][i, :n_points, :n_features]
            y_sample = self.data["y"][i, :n_points]
            
            # estimate noise as residual after linear fit (simple baseline)
            try:
                # add constant 1 to act as the intercept term
                X_with_intercept = np.column_stack([np.ones(n_points), X_sample])
                # solve least squares, find the best fitting coefficients
                coeffs, residuals, _, _ = np.linalg.lstsq(X_with_intercept, y_sample, rcond=None)
                # residuals is sum of squared residuals
                
                if len(residuals) > 0:
                    # average squared error per data point
                    residual_variance = residuals[0] / n_points
                    noise_estimates.append(np.sqrt(residual_variance))
                else:
                    # perfect fit or underdetermined case (when there are more features than datapoints) needs computing manually
                    predictions = X_with_intercept @ coeffs
                    # compute the residual variance
                    residual_variance = np.var(y_sample - predictions)
                    noise_estimates.append(np.sqrt(residual_variance))
            except:
                continue
        
        noise_stats = {}
        if noise_estimates:
            noise_estimates = np.array(noise_estimates)
            noise_stats["mean_noise_std"] = float(np.mean(noise_estimates))
            noise_stats["median_noise_std"] = float(np.median(noise_estimates))
            noise_stats["std_noise_std"] = float(np.std(noise_estimates))
            
            # R²: 1 - (noise_var / target_var)
            r2_estimates = []
            for i in sample_indices[:len(noise_estimates)]:
                n_points = self.data["num_datapoints"][i]
                y_sample = self.data["y"][i, :n_points]
                target_var = np.var(y_sample)
                if target_var > 0 and len(r2_estimates) < len(noise_estimates):
                    noise_var = noise_estimates[len(r2_estimates)] ** 2
                    r2 = 1 - (noise_var / target_var)
                    r2_estimates.append(max(0, r2))  # non-negative R²
            
            if r2_estimates:
                noise_stats["mean_linear_r2"] = float(np.mean(r2_estimates))
                noise_stats["median_linear_r2"] = float(np.median(r2_estimates))
        
        return noise_stats
    
    
    def generate_report(self) -> str:
        """Generate a comprehensive text report of the analysis.
        
        Returns:
            Formatted string report
        """
        report_lines = []
        report_lines.append("=" * 50)
        report_lines.append("REGRESSION DATA ANALYSIS REPORT")
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
        
        # target distribution
        report_lines.append("TARGET DISTRIBUTION")
        report_lines.append("-" * 50)
        target_stats = self.analyze_target_distribution()
        for key, val in target_stats.items():
            if isinstance(val, bool):
                report_lines.append(f"{key}: {val}")
            elif isinstance(val, (int, np.integer)):
                report_lines.append(f"{key}: {val}")
            else:
                report_lines.append(f"{key}: {val:.4f}")
        report_lines.append("")
        
        # feature distribution
        report_lines.append("FEATURE DISTRIBUTION")
        report_lines.append("-" * 50)
        feature_stats = self.analyze_feature_distributions()
        for key, val in feature_stats.items():
            if np.isinf(val):
                report_lines.append(f"{key}: inf")
            else:
                report_lines.append(f"{key}: {val:.4f}")
        report_lines.append("")
        
        # feature correlations
        report_lines.append("FEATURE REDUNDANCY (Collinearity)")
        report_lines.append("-" * 50)
        redundancy_stats = self.analyze_feature_redundancy()
        if redundancy_stats:
            for key, val in redundancy_stats.items():
                report_lines.append(f"{key}: {val:.4f}")
        else:
            report_lines.append("No redundancy data available")
        report_lines.append("")
        
        # feature-target relationships
        report_lines.append("FEATURE-TARGET RELATIONSHIPS")
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
        
        # target scale and deviation
        report_lines.append("TARGET SCALE AND DEVIATION")
        report_lines.append("-" * 50)
        deviation_stats = self.analyze_target_scale_and_deviation()
        for key, val in deviation_stats.items():
            if np.isinf(val):
                report_lines.append(f"{key}: inf")
            else:
                report_lines.append(f"{key}: {val:.4f}")
        report_lines.append("")
        
        # noise characteristics
        report_lines.append("NOISE CHARACTERISTICS (Linear Model Baseline)")
        report_lines.append("-" * 50)
        noise_stats = self.analyze_noise_characteristics()
        if noise_stats:
            for key, val in noise_stats.items():
                if np.isinf(val):
                    report_lines.append(f"{key}: inf")
                else:
                    report_lines.append(f"{key}: {val:.4f}")
        else:
            report_lines.append("No noise data available")
        report_lines.append("")
        
        report_lines.append("=" * 50)
        
        return "\n".join(report_lines)
    
    
def compare_priors(analyzer1: RegressionDataAnalyzer, analyzer2: RegressionDataAnalyzer, 
                   name1: str, name2: str) -> str:
    """Compare two different priors side by side.
    
    Args:
        analyzer1: First analyzer
        analyzer2: Second analyzer
        name1: Name of first prior
        name2: Name of second prior
        
    Returns:
        Comparison report string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"COMPARISON: {name1} vs {name2}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # compare target distributions
    target1 = analyzer1.analyze_target_distribution()
    target2 = analyzer2.analyze_target_distribution()
    
    report_lines.append("TARGET DISTRIBUTION COMPARISON")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Metric':<30} {name1:<20} {name2:<20} {'Diff':<15}")
    report_lines.append("-" * 80)
    for key in target1.keys():
        if isinstance(target1[key], (bool, np.bool_)):
            report_lines.append(f"{key:<30} {str(target1[key]):<20} {str(target2[key]):<20} {'-':<15}")
        elif isinstance(target1[key], (int, np.integer)):
            diff = target2[key] - target1[key]
            report_lines.append(f"{key:<30} {target1[key]:<20} {target2[key]:<20} {diff:<15}")
        else:
            diff = target2[key] - target1[key]
            report_lines.append(f"{key:<30} {target1[key]:<20.4f} {target2[key]:<20.4f} {diff:<15.4f}")
    report_lines.append("")
    
    # compare feature distributions
    feature1 = analyzer1.analyze_feature_distributions()
    feature2 = analyzer2.analyze_feature_distributions()
    
    report_lines.append("FEATURE DISTRIBUTION COMPARISON")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Metric':<30} {name1:<20} {name2:<20} {'Diff':<15}")
    report_lines.append("-" * 80)
    for key in feature1.keys():
        if np.isinf(feature1[key]) or np.isinf(feature2[key]):
            report_lines.append(f"{key:<30} {'inf':<20} {'inf':<20} {'-':<15}")
        else:
            diff = feature2[key] - feature1[key]
            report_lines.append(f"{key:<30} {feature1[key]:<20.4f} {feature2[key]:<20.4f} {diff:<15.4f}")
    report_lines.append("")
    
    # compare feature-target relationships
    rel1 = analyzer1.analyze_target_feature_relationships()
    rel2 = analyzer2.analyze_target_feature_relationships()
    
    report_lines.append("FEATURE-TARGET RELATIONSHIPS COMPARISON")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Metric':<30} {name1:<20} {name2:<20} {'Diff':<15}")
    report_lines.append("-" * 80)
    for key in rel1.keys():
        diff = rel2[key] - rel1[key]
        report_lines.append(f"{key:<30} {rel1[key]:<20.4f} {rel2[key]:<20.4f} {diff:<15.4f}")
    report_lines.append("")
    
    # compare mutual information
    mi1 = analyzer1.analyze_mutual_information()
    mi2 = analyzer2.analyze_mutual_information()
    
    if mi1 and mi2:
        report_lines.append("MUTUAL INFORMATION COMPARISON")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Metric':<30} {name1:<20} {name2:<20} {'Diff':<15}")
        report_lines.append("-" * 80)
        for key in mi1.keys():
            if key in mi2:
                diff = mi2[key] - mi1[key]
                report_lines.append(f"{key:<30} {mi1[key]:<20.4f} {mi2[key]:<20.4f} {diff:<15.4f}")
        report_lines.append("")
    
    # compare redundancy
    redundancy1 = analyzer1.analyze_feature_redundancy()
    redundancy2 = analyzer2.analyze_feature_redundancy()
    
    if redundancy1 and redundancy2:
        report_lines.append("FEATURE REDUNDANCY COMPARISON")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Metric':<30} {name1:<20} {name2:<20} {'Diff':<15}")
        report_lines.append("-" * 80)
        for key in redundancy1.keys():
            if key in redundancy2:
                diff = redundancy2[key] - redundancy1[key]
                report_lines.append(f"{key:<30} {redundancy1[key]:<20.4f} {redundancy2[key]:<20.4f} {diff:<15.4f}")
        report_lines.append("")
    
    # compare target scale and deviation
    complexity1 = analyzer1.analyze_target_scale_and_deviation()
    complexity2 = analyzer2.analyze_target_scale_and_deviation()
    
    report_lines.append("TARGET SCALE AND DEVIATION COMPARISON")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Metric':<30} {name1:<20} {name2:<20} {'Diff':<15}")
    report_lines.append("-" * 80)
    for key in complexity1.keys():
        if np.isinf(complexity1[key]) or np.isinf(complexity2[key]):
            report_lines.append(f"{key:<30} {'inf':<20} {'inf':<20} {'-':<15}")
        else:
            diff = complexity2[key] - complexity1[key]
            report_lines.append(f"{key:<30} {complexity1[key]:<20.4f} {complexity2[key]:<20.4f} {diff:<15.4f}")
    report_lines.append("")
    
    # compare noise
    noise1 = analyzer1.analyze_noise_characteristics()
    noise2 = analyzer2.analyze_noise_characteristics()
    
    if noise1 and noise2:
        report_lines.append("NOISE CHARACTERISTICS COMPARISON")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Metric':<30} {name1:<20} {name2:<20} {'Diff':<15}")
        report_lines.append("-" * 80)
        for key in noise1.keys():
            if key in noise2:
                if np.isinf(noise1[key]) or np.isinf(noise2[key]):
                    report_lines.append(f"{key:<30} {'inf':<20} {'inf':<20} {'-':<15}")
                else:
                    diff = noise2[key] - noise1[key]
                    report_lines.append(f"{key:<30} {noise1[key]:<20.4f} {noise2[key]:<20.4f} {diff:<15.4f}")
        report_lines.append("")
    
    # summary interpretation
    report_lines.append("KEY DIFFERENCES SUMMARY")
    report_lines.append("-" * 80)
    
    # target variability
    var_diff = target2["variance"] - target1["variance"]
    if abs(var_diff) > 0.1:
        report_lines.append(f"• Target variance: {name2} is {'more' if var_diff > 0 else 'less'} variable ({abs(var_diff):.4f})")
    
    # function variability
    if complexity1 and complexity2:
        var_diff = complexity2["mean_target_deviation"] - complexity1["mean_target_deviation"]
        if abs(var_diff) > 0.01:
            report_lines.append(f"• Function variability: {name2} has {'higher' if var_diff > 0 else 'lower'} within-function variance ({abs(var_diff):.4f})")
    
    # correlation strength
    if rel1 and rel2:
        corr_diff = rel2["pearson_mean_abs"] - rel1["pearson_mean_abs"]
        if abs(corr_diff) > 0.05:
            report_lines.append(f"• Feature-target correlation: {name2} has {'stronger' if corr_diff > 0 else 'weaker'} linear relationships ({abs(corr_diff):.4f})")
    
    # nonlinearity
    if rel1 and rel2:
        nonlin_diff = rel2["nonlinearity_score"] - rel1["nonlinearity_score"]
        if abs(nonlin_diff) > 0.05:
            report_lines.append(f"• Nonlinearity: {name2} exhibits {'more' if nonlin_diff > 0 else 'less'} nonlinear patterns ({abs(nonlin_diff):.4f})")
    
    # noise/R²
    if noise1 and noise2 and "mean_linear_r2" in noise1 and "mean_linear_r2" in noise2:
        r2_diff = noise2["mean_linear_r2"] - noise1["mean_linear_r2"]
        if abs(r2_diff) > 0.05:
            report_lines.append(f"• Linear R²: {name2} is {'easier' if r2_diff > 0 else 'harder'} to fit linearly ({name2} R²={noise2['mean_linear_r2']:.3f} vs {name1} R²={noise1['mean_linear_r2']:.3f})")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)