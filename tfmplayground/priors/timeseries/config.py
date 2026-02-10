"""Configuration for time series priors.

This module defines default hyperparameters for time series data generation,
following the same pattern as TabICL's prior_config.py.

Two types of hyperparameters:
- Fixed: Constant across all generated data
- Sampled: Varied per batch/group for diversity
"""

from __future__ import annotations


# =============================================================================
# Fixed Hyperparameters
# =============================================================================

DEFAULT_TS_FIXED_HP = {
    # Lag features
    "include_lags": True,
    "max_lags": 10,
    
    # Normalization
    "normalize": True,
    
    # Sequence constraints
    "min_seq_len": 50,
    
    # Forecasting
    "forecast_horizon": 1,
    
    # Feature processing
    "permute_features": False,  # Don't permute - order matters for time series
}


# =============================================================================
# Sampled Hyperparameters
# =============================================================================

# Note: These use the same meta-distribution pattern as TabICL.
# The actual sampling logic will be implemented in Step 4/5.
# For now, we define the structure.

DEFAULT_TS_SAMPLED_HP = {
    # ----- Trend -----
    "trend_type": {
        "distribution": "meta_choice",
        "choice_values": ["none", "linear", "quadratic"],
    },
    "trend_strength": {
        "distribution": "meta_uniform",
        "min": 0.0,
        "max": 2.0,
    },
    
    # ----- Seasonality -----
    "seasonal_period": {
        "distribution": "meta_choice",
        "choice_values": [None, 7, 12, 24, 52],  # None = no seasonality
    },
    "seasonal_amplitude": {
        "distribution": "meta_uniform",
        "min": 0.0,
        "max": 3.0,
    },
    
    # ----- Autoregression -----
    "ar_order": {
        "distribution": "meta_choice",
        "choice_values": [0, 1, 2, 3, 5],  # 0 = no AR
    },
    "ar_persistence": {
        "distribution": "meta_uniform",
        "min": 0.1,
        "max": 0.95,  # Sum of |coefficients|, must be < 1 for stationarity
    },
    
    # ----- Noise -----
    "noise_std": {
        "distribution": "meta_log_uniform",
        "min": 0.01,
        "max": 0.5,
    },
    
    # ----- SCM Structure -----
    # Default to 1 layer (simple linear), but allow more if needed
    "num_layers": {
        "distribution": "meta_choice",
        "choice_values": [1, 1, 1, 2],  # Weighted toward 1 (simple)
    },
    "hidden_dim": {
        "distribution": "meta_choice",
        "choice_values": [16, 32, 64],
    },
    "activation": {
        "distribution": "meta_choice",
        "choice_values": ["tanh", "relu", "identity"],
    },
}


# =============================================================================
# Prior Type Configurations
# =============================================================================

# Different preset configurations for common use cases

TS_PRIOR_PRESETS = {
    # Pure AR process, no trend or seasonality
    "ar": {
        "trend_type": "none",
        "seasonal_period": None,
        "ar_order": 2,
    },
    
    # Trending data with no seasonality
    "trend": {
        "trend_type": "linear",
        "seasonal_period": None,
        "ar_order": 1,
    },
    
    # Seasonal data with no trend
    "seasonal": {
        "trend_type": "none",
        "seasonal_period": 12,
        "ar_order": 1,
    },
    
    # Full mixture - samples all components randomly
    "mix_ts": {
        # Uses DEFAULT_TS_SAMPLED_HP as-is
    },
}
