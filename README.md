# TFM-Playground

This repository builds on nanotabpfn, an open-source tabular foundation model. The base repository provides a compact implementation of the TabPFNv2 architecture (nanoTabPFN), a training loop, and an evaluation pipeline.

**Our contribution** lives in [`tfmplayground/priors/`](tfmplayground/priors/): a unified prior interface that wraps multiple synthetic data generators, together with a complete experiment pipeline in [`tfmplayground/priors/experiments/`](tfmplayground/priors/experiments/) for systematically comparing priors across regression and classification tasks.

## Getting Started

```bash
git clone <repo-url> && cd TFM-Playground
pip install -e .
```

All experiment settings are in a single config file: [`tfmplayground/priors/experiments/config.yaml`](tfmplayground/priors/experiments/config.yaml).

## Our Contribution: Prior Interface & Experiments

### Unified Prior Interface

The `tfmplayground.priors` package provides a single CLI and Python API to generate data from any supported prior family. Data can be generated on-the-fly during training or dumped to HDF5 for offline use.

**CLI example** — generate an HDF5 prior dump:
```bash
python -m tfmplayground.priors --lib tabicl \
       --prior_type mix_scm \
       --num_batches 1000 --batch_size 4 \
       --min_features 3 --max_features 3 \
       --max_seq_len 50 --max_classes 3 \
       --save_path tabicl_4k_50x3.h5
```

**Python example** — load a dump or generate on-the-fly:
```python
from tfmplayground.priors import PriorDumpDataLoader, TabICLPriorDataLoader

# From a pre-generated dump
prior = PriorDumpDataLoader("tabicl_4k_50x3.h5", num_steps=20, batch_size=4, device="cpu")

# On-the-fly generation
prior = TabICLPriorDataLoader(
    num_steps=20, batch_size=4,
    num_datapoints_max=50, min_features=3, max_features=3,
    max_num_classes=3, device="cpu",
)
```

See [`prior_visualization.ipynb`](prior_visualization.ipynb) for more examples.

### Supported Priors

#### Regression

| Library | Prior types | Description |
|---|---|---|
| TICL | `ticl_gp`, `ticl_mlp` | Gaussian Process, Multi-Layer Perceptron |
| TabPFN | `tabpfn_mlp`, `tabpfn_gp`, `tabpfn_prior_bag` | MLP, Gaussian Process, Prior Bag (ensemble) |
| Real data | `real_default_targets`, `real_random_targets` | Original target column; any matching target column |

#### Classification

| Library | Prior types | Description |
|---|---|---|
| TICL | `ticl_classification_adapter` | Classification Adapter with MLP/GP base |
| TabPFN | `tabpfn_mlp`, `tabpfn_gp`, `tabpfn_prior_bag` | MLP, Gaussian Process, Prior Bag (ensemble) |
| TabICL | `tabicl_mlp_scm`, `tabicl_tree_scm`, `tabicl_mix_scm` | MLP SCM, Tree SCM, Mix SCM (ensemble of MLP and Tree) |
| TabForestPFN | `tabforest_forest`, `tabforest_neighbor`, `tabforest_cuts` | Decision Tree Forest, KNN Neighbor, Random Cuts |
| Real data | `real_default_targets`, `real_random_targets` | Original target column; any matching target column |

### Experiment Pipeline

The full experiment workflow is in [`tfmplayground/priors/experiments/`](tfmplayground/priors/experiments/). It consists of four steps:

1. **Generate data** — create `.h5` prior dumps for all or selected priors:
   ```bash
   python tfmplayground/priors/experiments/generate_data.py --mode classification --priors all
   ```

2. **Analyze data** — interactively inspect distributions, feature importance, etc.:
   ```bash
   python tfmplayground/priors/experiments/analyze_priors.py --mode classification
   ```

3. **Train models** — train one nanoTabPFN per prior (supports resuming):
   ```bash
   python tfmplayground/priors/experiments/train_models.py --problem_type classification --priors all --epochs 50 --steps 100
   ```

4. **Compare models** — generate comparison plots, metrics, and TabArena evaluation:
   ```bash
   python tfmplayground/priors/experiments/compare_models.py --problem_type classification
   ```

See the full experiment documentation in [`tfmplayground/priors/experiments/README.md`](tfmplayground/priors/experiments/README.md).

---

## Base Repository: nanoTabPFN

The sections below document the original TFM-Playground functionality that our work builds on.

<details>
<summary>nanoTabPFN usage and pretraining</summary>

### Inference

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tfmplayground import NanoTabPFNClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

clf = NanoTabPFNClassifier()
clf.fit(X_train, y_train)

prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))
```

### Architecture

`tfmplayground/model.py` contains the architecture in less than 250 lines of code. `tfmplayground/train.py` implements a simple training loop in under 100 lines.

### Pretraining

Download 100k pre-generated classification datasets from [here](https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/50x3_3_100k_classification.h5), then:
```bash
python pretrain_classification.py --epochs 80 --steps 25 --batchsize 50 --priordump 50x3_3_100k_classification.h5
```

For regression, download 1.28M tables from [here](https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/50x3_1280k_regression.h5) and run `python pretrain_regressor.py`.

</details>
