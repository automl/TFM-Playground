"""Configuration for time series priors.

This module defines default hyperparameters for time series data generation.

Two types of configuration:
- DEFAULT_TS_FIXED_HP: Default values used by ForecastPriorDataset
- TS_PRIOR_PRESETS: Presets for different temporal pattern types (for CLI)

Note: Actual hyperparameter sampling logic is in ForecastPriorDataset.
      See sample_group_hyperparameters() and sample_dataset_hyperparameters().
"""

from __future__ import annotations


# =============================================================================
# Default Fixed Hyperparameters
# =============================================================================
# These are used as defaults by ForecastPriorDataset

DEFAULT_TS_FIXED_HP = {
    # Lag features
    "include_lags": True,
    "max_lags": 5,
    
    # Sequence constraints
    "min_seq_len": 50,
    "max_seq_len": 1024,
    
    # Feature constraints
    "min_features": 2,
    "max_features": 100,
    
    # Train/test split ratios (temporal)
    "min_train_ratio": 0.5,
    "max_train_ratio": 0.9,
    
    # Batch structure
    "batch_size": 256,
    "batch_size_per_gp": 4,
}


# =============================================================================
# Prior Type Presets
# =============================================================================
# Used when specifying --prior_type in CLI

TS_PRIOR_PRESETS = {
    # Pure AR process - high autocorrelation, minimal trend/seasonality
    "ar": {
        "trend_prob": 0.1,
        "seasonal_prob": 0.1,
        "ar_prob": 0.9,
    },
    
    # Trending data - strong trends, some AR
    "trend": {
        "trend_prob": 0.8,
        "seasonal_prob": 0.1,
        "ar_prob": 0.4,
    },
    
    # Seasonal data - strong periodicity
    "seasonal": {
        "trend_prob": 0.1,
        "seasonal_prob": 0.8,
        "ar_prob": 0.4,
    },
    
    # Full mixture - random sampling of all patterns
    "mix_ts": {
        # Uses random group sampling (see ForecastPriorDataset)
    },
}
