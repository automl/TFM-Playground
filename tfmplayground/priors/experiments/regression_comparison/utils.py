"""Shared utilities for experiments."""

from typing import Dict

import yaml


def load_config(config_path: str = "config.yaml") -> Dict:
    """load configuration from YAML file."""

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
