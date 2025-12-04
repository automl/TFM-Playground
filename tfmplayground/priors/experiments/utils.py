"""Shared utilities for experiments.

Includes:
- Config loading
- HDF5 prior discovery
- Analyzer loading
- Plot styling and saving
"""

import itertools
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


# config & file discovery
def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def discover_h5_files(data_dir: str) -> Dict[str, str]:
    """Discover all HDF5 files in the data directory.

    Scans for files matching pattern: prior_*.h5
    Extracts prior name from filename.

    Args:
        data_dir: Directory containing .h5 data files.

    Returns:
        Dictionary mapping prior names to file paths.
        Example: {'ticl_gp': '/path/to/prior_ticl_gp_100x8.h5', ...}
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        return {}

    discovered: Dict[str, str] = {}

    for filepath in data_path.glob("prior_*.h5"):
        # extract prior name from filename: prior_<name>_<params>.h5
        filename = filepath.stem  # Remove .h5

        # remove 'prior_' prefix
        name_part = filename[6:] if filename.startswith("prior_") else filename

        # prior name is everything before the first digit (parameters start)
        match = re.match(r"^([a-zA-Z_]+)", name_part)
        if match:
            prior_name = match.group(1).rstrip("_")
            discovered[prior_name] = str(filepath)

    return discovered


# loading the analyzers for plots
def load_multiple_analyzers(
    data_dir: str,
    analyzer_class: Any,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Load all available prior analyzers.

    Scans data_dir for .h5 files and creates analyzer objects for each.
    Handles errors gracefully – skips files that fail to load.

    Args:
        data_dir: Directory containing .h5 data files.
        analyzer_class: RegressionDataAnalyzer or ClassificationDataAnalyzer.
        verbose: Print loading progress.

    Returns:
        Dictionary mapping prior names to analyzer objects.
        Example: {'ticl_gp': <RegressionDataAnalyzer>, ...}
    """
    if verbose:
        print(f"Scanning {data_dir} for prior data files...")

    discovered_files = discover_h5_files(data_dir)

    if not discovered_files:
        raise ValueError(
            f"No prior data files found in {data_dir}. "
            "Generate some data first using data_generation.py"
        )

    if verbose:
        print(f"   Found {len(discovered_files)} prior(s): {list(discovered_files.keys())}")
        print("\n Loading analyzers...")

    analyzers: Dict[str, Any] = {}
    failed: List[Tuple[str, str]] = []

    for prior_name, filepath in discovered_files.items():
        try:
            analyzer = analyzer_class(filepath)
            analyzers[prior_name] = analyzer
            if verbose:
                print(f"   [OK] {prior_name}")
        except Exception as e:
            failed.append((prior_name, str(e)))
            if verbose:
                print(f"   [WARNING] {prior_name} - FAILED: {e}")

    if not analyzers:
        error_msg = "No priors loaded successfully!\n"
        for name, err in failed:
            error_msg += f"  - {name}: {err}\n"
        raise ValueError(error_msg)

    if verbose:
        print(f"\n[OK] Loaded {len(analyzers)}/{len(discovered_files)} prior(s)")
        if failed:
            print(f"[WARNING] Failed to load {len(failed)} prior(s): {[f[0] for f in failed]}")

    return analyzers


# plotting utilities
def get_prior_colors(prior_names: List[str]) -> Dict[str, str]:
    """Map prior names to colors using tab10, cycling if there are >10 priors."""
    base_colors = plt.cm.tab10.colors

    # cycle through the 10 base colors for however many priors we have
    color_iter = itertools.cycle(base_colors)
    colors = [next(color_iter) for _ in prior_names]

    hex_colors = [mpl.colors.rgb2hex(c) for c in colors]
    return dict(zip(prior_names, hex_colors))


@contextmanager
def apply_plot_style():
    """Context manager for consistent plot styling with clean aesthetics."""
    original_rc = plt.rcParams.copy()

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.2,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "patch.linewidth": 1.0,
            "font.family": "sans-serif",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#CCCCCC",
        }
    )
    try:
        yield
    finally:
        plt.rcParams.update(original_rc)


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: str = "./results/figures",
) -> str:
    """Save figure as PNG.

    Args:
        fig: Matplotlib figure.
        filename: Filename without extension.
        output_dir: Output directory.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / f"{filename}.png"
    fig.savefig(filepath, dpi=300, bbox_inches="tight")

    return str(filepath)


def save_all_figures(
    figures: List[Tuple[plt.Figure, str]],
    output_dir: str = "./results/figures",
) -> None:
    """Save multiple figures.

    Args:
        figures: List of (figure, filename) tuples.
        output_dir: Output directory.
    """
    print(f"\nSaving {len(figures)} figure(s) to {output_dir}")

    for fig, filename in figures:
        try:
            save_figure(fig, filename, output_dir)
            print(f"   [OK] {filename}")
        except Exception as e:  # noqa: BLE001
            print(f"   [WARNING] {filename} - FAILED: {e}")