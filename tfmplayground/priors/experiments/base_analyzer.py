"""Base analyzer for datasets generated from synthetic priors."""

import numbers
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy import stats

from tfmplayground.priors.experiments.utils.general import load_config


class DataAnalyzer(ABC):
    """Base analyzer for datasets generated from synthetic priors.
    
    Provides shared loading and basic statistics utilities for regression and
    classification analyzers.
    """

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        config = load_config(str(Path(__file__).resolve().parent / "config.yaml"))
        self.analysis_config = config["analysis"]
        self.sample_size = self.analysis_config["analyzer_sample_size"]
        self.random_state = self.analysis_config["random_state"]

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

        total_samples = len(self.data["X"])
        if 0 < self.sample_size < total_samples:
            # read the HDF5 arrays sequentially first, then subsample in memory.
            # this was faster than asking
            # h5py for many random rows, ofc assuming the full read fits in RAM.
            rng = np.random.default_rng(self.random_state)
            idx = np.sort(
                rng.choice(total_samples, size=self.sample_size, replace=False)
            )
            for key in self.data:
                self.data[key] = self.data[key][idx]
            print(f"  Loaded {self.sample_size} / {total_samples} samples")
        else:
            print(f"  Loaded {total_samples} samples")

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


    def analyze_feature_distributions(self) -> Dict[str, Any]:
        """Analyze the distribution of feature values.

        Returns:
            Dictionary with feature distribution statistics
        """
        features = []
        for i in range(len(self.data["X"])):
            n_points = self.data["num_datapoints"][i]
            n_features = self.data["num_features"][i]
            features.append(self.data["X"][i, :n_points, :n_features].ravel())

        if not features:
            return {}

        features = np.concatenate(features)

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


    def analyze_feature_redundancy(self) -> Dict[str, Any]:
        """Analyze redundancy and collinearity among features.

        Returns:
            Dictionary with redundancy metrics
        """

        all_corr = []
        high_corr_pair_counts = []
        
        for i in range(len(self.data["X"])):
            n_points = self.data["num_datapoints"][i]
            n_features = self.data["num_features"][i]
            
            # skip the cases with too few features
            if n_features < 2:
                continue
            
            X_sample = self.data["X"][i, :n_points, :n_features]
            
            # compute the correlation matrix
            try:
                # Drop constant (zero-variance) columns to avoid divide-by-zero warnings in corrcoef
                std = np.nanstd(X_sample, axis=0)
                non_constant = std > 0
                X_nc = X_sample[:, non_constant]

                # Need at least two non-constant features to compute pairwise correlations
                if X_nc.shape[1] < 2:
                    continue

                # this is of shape n_features x n_features
                corr_matrix = np.corrcoef(X_nc.T)

                # get upper triangle (excluding the diagonal aka correlation of feature with itself)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                correlations = corr_matrix[mask]
                # now it has unique feature pair correlations flattened into a 1D array

                # filter out nan/inf correlations (can still happen in degenerate edge cases)
                correlations = correlations[np.isfinite(correlations)]
                if correlations.size == 0:
                    continue

                all_corr.extend(correlations)

                # measure the high correlation pairs if corr coefficient is > 0.8
                high_corr = np.abs(correlations) > 0.8
                # collect the number of highly correlated pairs
                high_corr_pair_counts.append(high_corr.sum())
            except Exception:
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

    def prior_summary_vector(self) -> Tuple[np.ndarray, List[str]]:
        """Build the analyzer-specific summary vector used for prior similarity."""
        raise NotImplementedError("Subclasses must implement prior_summary_vector().")


# prior similarity computations

def compute_summary_similarity_matrix(summary_matrix: np.ndarray) -> np.ndarray:
    """Convert summary vectors into a distance-based similarity matrix.

    Steps:
    1. z-score each summary metric across priors
    2. compute pairwise Euclidean distances
    3. turn distances into similarities with an RBF kernel
    """
    summary_matrix = np.asarray(summary_matrix, dtype=float)
    if summary_matrix.ndim != 2:
        raise ValueError("summary_matrix must be 2-D")
    if summary_matrix.shape[0] == 1:
        return np.array([[1.0]], dtype=float)

    mean = summary_matrix.mean(axis=0, keepdims=True)
    std = summary_matrix.std(axis=0, keepdims=True)
    summary_z = (summary_matrix - mean) / np.where(std > 1e-8, std, 1.0)

    diffs = summary_z[:, None, :] - summary_z[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2))

    tri = np.triu_indices_from(distances, k=1)
    nonzero_distances = distances[tri]
    nonzero_distances = nonzero_distances[
        np.isfinite(nonzero_distances) & (nonzero_distances > 0)
    ]

    if nonzero_distances.size == 0:
        return np.ones_like(distances, dtype=float)

    sigma = float(np.median(nonzero_distances))
    if not np.isfinite(sigma) or sigma <= 1e-8:
        sigma = float(np.mean(nonzero_distances))
    if not np.isfinite(sigma) or sigma <= 1e-8:
        sigma = 1.0

    sim_matrix = np.exp(-(distances ** 2) / (2.0 * sigma ** 2))
    sim_matrix = np.clip(
        np.nan_to_num(sim_matrix, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0,
    )
    np.fill_diagonal(sim_matrix, 1.0)
    return sim_matrix


def compute_prior_similarity_matrix(
    analyzers: Dict[str, DataAnalyzer],
) -> Tuple[List[str], np.ndarray]:
    """Compute prior-prior similarity from analyzer summary vectors."""
    prior_names = list(analyzers.keys())

    summary_vectors = []
    for name in prior_names:
        vec, _ = analyzers[name].prior_summary_vector()
        summary_vectors.append(vec)

    summary_matrix = np.vstack(summary_vectors)
    if summary_matrix.shape[0] == 1:
        return prior_names, np.array([[1.0]], dtype=float)

    return prior_names, compute_summary_similarity_matrix(summary_matrix)
