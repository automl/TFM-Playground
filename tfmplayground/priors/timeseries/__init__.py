"""Time series priors for TFM-Playground.

This module provides synthetic data generation with temporal structure
for training tabular foundation models on forecasting tasks.

Main components:
- TemporalXSampler: Generates time-correlated features (trends, seasonality, AR)
- TimeSeriesSCM: Structural Causal Model respecting temporal causality
- ForecastPriorDataset: Dataset class with temporal train/test splits
"""

from .temporal_sampler import TemporalXSampler
from .timeseries_scm import TimeSeriesSCM
from .forecast_dataset import ForecastPriorDataset
from .config import DEFAULT_TS_FIXED_HP, TS_PRIOR_PRESETS

__all__ = [
    "TemporalXSampler",
    "TimeSeriesSCM",
    "ForecastPriorDataset",
    "DEFAULT_TS_FIXED_HP",
    "TS_PRIOR_PRESETS",
]
