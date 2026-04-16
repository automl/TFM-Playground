"""Vendored generators from TabForestPFN (den Breejen et al.).

Source: https://github.com/FelixDenBreeworked/TabForestPFN
These generators produce classification-only synthetic tabular data
using various decision-boundary strategies.
"""

from .synthetic_generator_forest import (
    synthetic_dataset_generator_forest,
    synthetic_dataset_function_forest,
)
from .synthetic_generator_neighbor import (
    synthetic_dataset_generator_neighbor,
    synthetic_dataset_function_neighbor,
)
from .synthetic_generator_cuts import (
    synthetic_dataset_generator_cut,
    synthetic_dataset_function_cuts,
)

__all__ = [
    "synthetic_dataset_generator_forest",
    "synthetic_dataset_function_forest",
    "synthetic_dataset_generator_neighbor",
    "synthetic_dataset_function_neighbor",
    "synthetic_dataset_generator_cut",
    "synthetic_dataset_function_cuts",
]
