"""Experiment utilities package.

Submodules:
    general       – prior discovery, config loading, plotting helpers
    training      – model training, metrics payload building
    visualization – comparison plots, decision boundaries, etc.
"""

from utils.general import discover_h5_files, load_config
from utils.training import train_model, _build_metrics_payload, _json_safe
