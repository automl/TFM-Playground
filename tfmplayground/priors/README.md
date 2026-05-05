# Priors

This package provides a unified interface for generating synthetic tabular data from multiple prior families. It supports data generation via CLI, on-the-fly loading during training, and dumping to HDF5 for offline use.

## Files

- `config.py`: default configurations for available priors
- `dataloader.py`: unified data loaders — `PriorDumpDataLoader` for HDF5 files, plus per-library loaders (`TICLPriorDataLoader`, `TabPFNPriorDataLoader`, `TabICLPriorDataLoader`, `TabForestPFNPriorDataLoader`)
- `utils.py`: builder functions (`build_ticl_prior`, `build_tabpfn_prior`) and `dump_prior_to_h5`
- `main.py`: CLI entry point (`python -m tfmplayground.priors`)

## Subfolders

- `experiments/`: scripts for systematic prior comparison (generate, train, compare). See [experiments/README.md](experiments/README.md).
- `real_data/`: real-data prior pipeline using cached OpenML datasets. See [real_data/README.md](real_data/README.md).
- `vendors/`: vendored third-party code. Contains `tabforestpfn/` because the TabForestPFN repository is not installable as a package — the relevant generator modules are included directly.

## Supported Priors

### Regression

| Library | Prior types | Description |
|---|---|---|
| TICL | `ticl_gp`, `ticl_mlp` | Gaussian Process, Multi-Layer Perceptron |
| TabPFN | `tabpfn_mlp`, `tabpfn_gp`, `tabpfn_prior_bag` | MLP, Gaussian Process, Prior Bag (ensemble) |
| Real data | `real_default_targets`, `real_random_targets` | Original target column; any matching target column |

### Classification

| Library | Prior types | Description |
|---|---|---|
| TICL | `ticl_classification_adapter` | Classification Adapter with MLP/GP base |
| TabPFN | `tabpfn_mlp`, `tabpfn_gp`, `tabpfn_prior_bag` | MLP, Gaussian Process, Prior Bag (ensemble) |
| TabICL | `tabicl_mlp_scm`, `tabicl_tree_scm`, `tabicl_mix_scm` | MLP SCM, Tree SCM, Mix SCM (ensemble of MLP and Tree) |
| TabForestPFN | `tabforest_forest`, `tabforest_neighbor`, `tabforest_cuts` | Decision Tree Forest, KNN Neighbor, Random Cuts |
| Real data | `real_default_targets`, `real_random_targets` | Original target column; any matching target column |

## CLI Usage

```bash
python -m tfmplayground.priors \
  --lib tabicl \
  --prior_type mix_scm \
  --num_batches 1000 --batch_size 4 \
  --min_features 3 --max_features 3 \
  --max_seq_len 50 --max_classes 3 \
  --save_path tabicl_4k_50x3.h5
```

Task type is inferred from `--max_classes`: `> 0` for classification, `0` for regression.

## Python Usage

Load a pre-generated HDF5 dump:

```python
from tfmplayground.priors import PriorDumpDataLoader

prior = PriorDumpDataLoader("tabicl_4k_50x3.h5", num_steps=20, batch_size=4, device="cpu")
```

Or generate on-the-fly:

```python
from tfmplayground.priors import TabICLPriorDataLoader

prior = TabICLPriorDataLoader(
    num_steps=20, batch_size=4,
    num_datapoints_max=50, min_features=3, max_features=3,
    max_num_classes=3, device="cpu",
)
```
