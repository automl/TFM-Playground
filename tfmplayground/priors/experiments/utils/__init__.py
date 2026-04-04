"""Experiment utilities package.

Submodules:
    general       – prior discovery, config loading, plotting helpers
    training      – model training, metrics payload building
    visualization – comparison plots, decision boundaries, etc.
"""

from .general import (
    discover_h5_files,
    load_config,
    get_prior_colors,
    apply_plot_style,
    merge_variable_width_features,
)
from .training import train_model, _build_metrics_payload, _json_safe
from .visualization import (
    plot_tabarena_performance_heatmap,
    plot_prior_correlation_heatmap,
)

__all__ = [
    "discover_h5_files",
    "load_config",
    "get_prior_colors",
    "apply_plot_style",
    "merge_variable_width_features",
    "train_model",
    "_build_metrics_payload",
    "_json_safe",
    "plot_tabarena_performance_heatmap",
    "plot_prior_correlation_heatmap",
]
