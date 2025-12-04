"""Base analyzer for datasets generated from synthetic priors."""

import numbers
from abc import ABC
from typing import Any, Dict, Optional

import h5py
import numpy as np
from scipy import stats


class DataAnalyzer(ABC):
    """Base analyzer for datasets generated from synthetic priors.
    
    Provides shared loading and basic statistics utilities for regression and
    classification analyzers.
    """

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.data: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Any] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Load common dataset structure from an HDF5 file.

        Expects groups/arrays `X`, `y`, `num_features`, `num_datapoints`,
        and `single_eval_pos`. Optionally reads `problem_type` metadata if
        present.
        """
        print(f"Loading data from {self.h5_path}...")

        with h5py.File(self.h5_path, "r") as f:
            self.data = {
                "X": f["X"][:],
                "y": f["y"][:],
                "num_features": f["num_features"][:],
                "num_datapoints": f["num_datapoints"][:],
                "single_eval_pos": f["single_eval_pos"][:],
            }

            if "problem_type" in f:
                raw = f["problem_type"][()]
                if isinstance(raw, bytes):
                    self.metadata["problem_type"] = raw.decode("utf-8")
                else:
                    self.metadata["problem_type"] = str(raw)

        print(f"  Loaded {len(self.data['X'])} samples")
        if "problem_type" in self.metadata:
            print(f"  Problem type: {self.metadata['problem_type']}")

    def get_basic_statistics(self) -> Dict[str, Any]:
        """Extract basic statistics of the dataset common to all analyzers.

        Returns:
            Dictionary containing various statistics.
        """
        stats_dict: Dict[str, Any] = {
            "total_samples": len(self.data["X"]),
            # samples could potentially have different lengths/features
            "max_seq_len": self.data["X"].shape[1],
            "max_features": self.data["X"].shape[2],
        }

        # actual sequence lengths (because it's padded to maximums)
        stats_dict["seq_lengths"] = {
            "min": int(self.data["num_datapoints"].min()),
            "max": int(self.data["num_datapoints"].max()),
            "mean": float(self.data["num_datapoints"].mean()),
            "std": float(self.data["num_datapoints"].std()),
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

    @staticmethod
    def _format_stat_line(key: str, val: Any) -> Optional[str]:
        """Format a single key/value pair as a line for text reports.
        
        Skips array-like values (np.ndarray, list, tuple) and handles bool, int, 
        float, inf, and nan values.

        Args:
            key: Name of the metric or field.
            val: Value to format.

        Returns:
            Formatted line as a string, or None if the value should be skipped.
        """
        # skip array-ish things:
        if isinstance(val, (np.ndarray, list, tuple)):
            return None

        # booleans
        if isinstance(val, (bool, np.bool_)):  # type: ignore[attr-defined]
            return f"{key}: {val}"

        # numeric values
        if isinstance(val, numbers.Number):
            v = float(val)
            if np.isinf(v):
                return f"{key}: inf"
            if np.isnan(v):
                return f"{key}: nan"
            return f"{key}: {v:.4f}"

        # fallback: string or other printable stuff
        return f"{key}: {val}"


    def analyze_feature_distributions(self, sample_size: int = 100) -> Dict[str, Any]:
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

        feature_stats: Dict[str, Any] = {
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


    def analyze_feature_redundancy(self, n_samples: int = 100) -> Dict[str, Any]:
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
            redundancy_stats: Dict[str, Any] = {
                "mean_abs_correlation": float(np.abs(all_corr).mean()),
                "max_abs_correlation": float(np.abs(all_corr).max()),
                "std_correlation": float(all_corr.std()),
                "high_correlation_ratio": float((np.abs(all_corr) > 0.8).sum() / len(all_corr)),
                "mean_high_corr_pairs": float(np.mean(high_corr_pair_counts)) if high_corr_pair_counts else 0.0,
            }
        else:
            redundancy_stats = {}
        
        return redundancy_stats