"""Forecast Prior Dataset for batch generation with temporal splits.

This module provides ForecastPriorDataset, which wraps TimeSeriesSCM
and handles batch generation with proper temporal train/test splits.

Unlike TabICL's random splits, this ensures training data is always
"in the past" relative to test data.
"""

from __future__ import annotations

import random
from typing import Tuple, Optional, Dict, Any, Union, List

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from .timeseries_scm import TimeSeriesSCM
from .config import DEFAULT_TS_FIXED_HP, TS_PRIOR_PRESETS


class ForecastPriorDataset(IterableDataset):
    """Dataset for generating batches of synthetic time series data.
    
    Wraps TimeSeriesSCM and provides:
    - Batch generation with configurable batch size
    - Temporal train/test splits (no future leakage)
    - Hyperparameter sampling for diverse data
    - Padding for variable number of features
    
    Parameters
    ----------
    batch_size : int, default=256
        Number of datasets per batch.
    
    batch_size_per_gp : int, default=4
        Number of datasets per group (share temporal pattern type).
    
    min_features : int, default=2
        Minimum number of features.
    
    max_features : int, default=100
        Maximum number of features.
    
    min_seq_len : int, optional
        Minimum sequence length. If None, uses max_seq_len.
    
    max_seq_len : int, default=1024
        Maximum sequence length.
    
    min_train_ratio : float, default=0.5
        Minimum fraction of data for training.
    
    max_train_ratio : float, default=0.9
        Maximum fraction of data for training.
    
    include_lags : bool, default=True
        Whether to include lag features.
    
    max_lags : int, default=5
        Maximum number of lag features per base feature.
    
    prior_type : str, default="mix_ts"
        Type of temporal patterns: "ar", "trend", "seasonal", or "mix_ts".
    
    fixed_hp : dict, optional
        Override default fixed hyperparameters.
    
    device : str, default="cpu"
        Device to place tensors on.
    
    Examples
    --------
    >>> dataset = ForecastPriorDataset(batch_size=32, max_seq_len=100)
    >>> batch = dataset.get_batch()
    >>> batch["x"].shape
    torch.Size([32, 100, ...])
    """
    
    def __init__(
        self,
        batch_size: Optional[int] = None,
        batch_size_per_gp: Optional[int] = None,
        min_features: Optional[int] = None,
        max_features: Optional[int] = None,
        min_seq_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        min_train_ratio: Optional[float] = None,
        max_train_ratio: Optional[float] = None,
        include_lags: Optional[bool] = None,
        max_lags: Optional[int] = None,
        prior_type: str = "mix_ts",
        fixed_hp: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        super().__init__()
        
        # Merge provided fixed_hp with defaults
        hp = {**DEFAULT_TS_FIXED_HP, **(fixed_hp or {})}
        
        # Use provided values or fall back to config defaults
        self.batch_size = batch_size if batch_size is not None else hp.get("batch_size", 256)
        self.batch_size_per_gp = batch_size_per_gp if batch_size_per_gp is not None else hp.get("batch_size_per_gp", 4)
        self.min_features = min_features if min_features is not None else hp.get("min_features", 2)
        self.max_features = max_features if max_features is not None else hp.get("max_features", 100)
        self.min_seq_len = min_seq_len if min_seq_len is not None else hp.get("min_seq_len", 50)
        self.max_seq_len = max_seq_len if max_seq_len is not None else hp.get("max_seq_len", 1024)
        self.min_train_ratio = min_train_ratio if min_train_ratio is not None else hp.get("min_train_ratio", 0.5)
        self.max_train_ratio = max_train_ratio if max_train_ratio is not None else hp.get("max_train_ratio", 0.9)
        self.include_lags = include_lags if include_lags is not None else hp.get("include_lags", True)
        self.max_lags = max_lags if max_lags is not None else hp.get("max_lags", 5)
        self.prior_type = prior_type
        self.device = device
        
        # Load preset if using a specific prior type
        self.preset = TS_PRIOR_PRESETS.get(prior_type, {})
    
    def get_batch(self, batch_size: Optional[int] = None) -> Dict[str, Union[Tensor, int]]:
        """Generate a batch of synthetic time series datasets.
        
        Parameters
        ----------
        batch_size : int, optional
            Override default batch size.
        
        Returns
        -------
        batch : dict
            Dictionary containing:
            - "x": Features tensor (batch_size, seq_len, num_features)
            - "y": Target tensor (batch_size, seq_len)
            - "target_y": Same as y (for compatibility)
            - "single_eval_pos": Temporal split position (int)
        """
        batch_size = batch_size or self.batch_size
        
        # Sample sequence length (fixed for this batch)
        seq_len = self.sample_seq_len()
        
        # Sample temporal split position
        split_pos = self.sample_temporal_split(seq_len)
        
        # Calculate number of groups
        num_groups = (batch_size + self.batch_size_per_gp - 1) // self.batch_size_per_gp
        
        X_list: List[Tensor] = []
        y_list: List[Tensor] = []
        feature_counts: List[int] = []
        
        dataset_idx = 0
        
        for group_idx in range(num_groups):
            # Sample group-level hyperparameters (shared within group)
            group_hp = self.sample_group_hyperparameters()
            
            # How many datasets in this group
            group_size = min(self.batch_size_per_gp, batch_size - dataset_idx)
            
            for _ in range(group_size):
                # Sample dataset-specific hyperparameters
                dataset_hp = self.sample_dataset_hyperparameters(group_hp)
                
                # Create SCM and generate data
                scm = TimeSeriesSCM(
                    seq_len=seq_len,
                    num_features=dataset_hp["num_features"],
                    num_outputs=1,
                    include_lags=self.include_lags,
                    max_lags=self.max_lags,
                    num_layers=dataset_hp["num_layers"],
                    hidden_dim=dataset_hp["hidden_dim"],
                    activation=dataset_hp["activation"],
                    noise_std=dataset_hp["noise_std"],
                    device=self.device,
                    trend_prob=group_hp["trend_prob"],
                    seasonal_prob=group_hp["seasonal_prob"],
                    ar_prob=group_hp["ar_prob"],
                )
                
                X, y = scm()
                
                X_list.append(X)
                y_list.append(y)
                feature_counts.append(X.shape[1])
                
                dataset_idx += 1
        
        # Pad features to max feature count in this batch
        max_features_in_batch = max(feature_counts)
        X_padded = self._pad_features(X_list, max_features_in_batch, seq_len)
        
        # Stack into batch tensors
        X_batch = torch.stack(X_padded)  # (batch_size, seq_len, max_features)
        y_batch = torch.stack(y_list)    # (batch_size, seq_len)
        
        return {
            "x": X_batch.to(self.device),
            "y": y_batch.to(self.device),
            "target_y": y_batch.to(self.device),
            "single_eval_pos": split_pos,
        }
    
    def _pad_features(
        self, 
        X_list: List[Tensor], 
        max_features: int, 
        seq_len: int
    ) -> List[Tensor]:
        """Pad feature tensors to have the same number of features.
        
        Parameters
        ----------
        X_list : list of Tensor
            List of feature tensors with varying feature counts.
        
        max_features : int
            Target number of features (pad to this).
        
        seq_len : int
            Sequence length.
        
        Returns
        -------
        X_padded : list of Tensor
            List of padded tensors, all with shape (seq_len, max_features).
        """
        X_padded = []
        for X in X_list:
            current_features = X.shape[1]
            if current_features < max_features:
                padding = torch.zeros(
                    seq_len, 
                    max_features - current_features, 
                    device=self.device
                )
                X = torch.cat([X, padding], dim=1)
            X_padded.append(X)
        return X_padded
    
    def sample_seq_len(self) -> int:
        """Sample sequence length for this batch.
        
        Returns
        -------
        seq_len : int
            Sampled sequence length.
        """
        if self.min_seq_len is None:
            return self.max_seq_len
        return random.randint(self.min_seq_len, self.max_seq_len)
    
    def sample_temporal_split(self, seq_len: int) -> int:
        """Sample a temporal train/test split position.
        
        Unlike random splits, this ensures training data comes
        before test data in time.
        
        Parameters
        ----------
        seq_len : int
            Total sequence length.
        
        Returns
        -------
        split_pos : int
            Position separating train (past) from test (future).
        """
        train_ratio = random.uniform(self.min_train_ratio, self.max_train_ratio)
        split_pos = int(seq_len * train_ratio)
        
        # Ensure at least some data on both sides
        split_pos = max(10, min(split_pos, seq_len - 10))
        
        return split_pos
    
    def sample_group_hyperparameters(self) -> Dict[str, Any]:
        """Sample hyperparameters shared within a group.
        
        Groups share the same temporal pattern type (trending, seasonal, etc.)
        If a preset is set (via prior_type), uses those fixed values.
        
        Returns
        -------
        hp : dict
            Group-level hyperparameters.
        """
        # If using a preset (ar, trend, seasonal), use fixed values
        if self.preset:
            return {
                "pattern_type": self.prior_type,
                "trend_prob": self.preset.get("trend_prob", 0.3),
                "seasonal_prob": self.preset.get("seasonal_prob", 0.3),
                "ar_prob": self.preset.get("ar_prob", 0.5),
            }
        
        # Otherwise (mix_ts), randomly sample pattern type
        pattern_type = random.choice(["trending", "seasonal", "ar_heavy", "mixed"])
        
        if pattern_type == "trending":
            trend_prob = random.uniform(0.6, 0.9)
            seasonal_prob = random.uniform(0.0, 0.3)
            ar_prob = random.uniform(0.3, 0.6)
        elif pattern_type == "seasonal":
            trend_prob = random.uniform(0.0, 0.3)
            seasonal_prob = random.uniform(0.6, 0.9)
            ar_prob = random.uniform(0.3, 0.6)
        elif pattern_type == "ar_heavy":
            trend_prob = random.uniform(0.0, 0.3)
            seasonal_prob = random.uniform(0.0, 0.3)
            ar_prob = random.uniform(0.7, 0.95)
        else:  # mixed
            trend_prob = random.uniform(0.2, 0.5)
            seasonal_prob = random.uniform(0.2, 0.5)
            ar_prob = random.uniform(0.3, 0.7)
        
        return {
            "pattern_type": pattern_type,
            "trend_prob": trend_prob,
            "seasonal_prob": seasonal_prob,
            "ar_prob": ar_prob,
        }
    
    def sample_dataset_hyperparameters(self, group_hp: Dict[str, Any]) -> Dict[str, Any]:
        """Sample hyperparameters for a single dataset.
        
        Parameters
        ----------
        group_hp : dict
            Group-level hyperparameters to inherit from.
        
        Returns
        -------
        hp : dict
            Dataset-specific hyperparameters.
        """
        return {
            # Feature count varies per dataset
            "num_features": random.randint(self.min_features, self.max_features),
            
            # SCM structure (Option C: mostly 1 layer)
            "num_layers": random.choices([1, 2], weights=[0.8, 0.2])[0],
            "hidden_dim": random.choice([16, 32, 64]),
            "activation": random.choice(["tanh", "relu", "identity"]),
            
            # Noise level
            "noise_std": random.uniform(0.05, 0.3),
        }
    
    def __iter__(self):
        """Infinite iterator over batches."""
        return self
    
    def __next__(self) -> Dict[str, Union[Tensor, int]]:
        """Generate next batch."""
        return self.get_batch()
    
    def __repr__(self) -> str:
        return (
            f"ForecastPriorDataset(\n"
            f"  prior_type={self.prior_type},\n"
            f"  batch_size={self.batch_size},\n"
            f"  batch_size_per_gp={self.batch_size_per_gp},\n"
            f"  features={self.min_features}-{self.max_features},\n"
            f"  seq_len={self.min_seq_len}-{self.max_seq_len},\n"
            f"  train_ratio={self.min_train_ratio}-{self.max_train_ratio},\n"
            f"  include_lags={self.include_lags},\n"
            f"  max_lags={self.max_lags},\n"
            f"  device={self.device}\n"
            f")"
        )
