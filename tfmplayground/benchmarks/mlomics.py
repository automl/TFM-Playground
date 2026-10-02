"""MLOmics TCGA classification benchmark, prepared exactly as in TabPFN-Wide
(analysis/load_mm_data.py and analysis/data.py), so that the folds are identical and results can
be compared fold by fold with their published per-fold results.

Data: https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark (Main_Dataset from Hugging Face,
AIBIC/MLOmics), expected at <root>/Main_Dataset/Classification_datasets/GS-<DATASET>/Original/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder

DATASETS = ("BRCA", "COAD", "GBM", "LGG", "OV")
OMIC_FILES = {"mrna": "mRNA", "cnv": "CNV", "methylation": "Methy", "mirna": "miRNA"}
# Feature counts evaluated by TabPFN-Wide; 0 means all features
N_FEATURES = (200, 500, 2000, 5000, 7500, 10000, 15000, 20000, 25000, 30000, 0)
N_SPLITS, N_REPEATS, SEED = 5, 1, 42


def _load_omic(path: Path) -> pd.DataFrame:
    """One omic file (features x samples) -> samples x features, cleaned as TabPFN-Wide does."""
    data = pd.read_csv(path, index_col=0)
    data = data.loc[~(data == 0).all(axis=1)]  # drop all-zero features
    data = data[~data.index.duplicated(keep="first")]  # drop duplicated features
    data = data.T
    data = data.fillna(data.mean())
    data.columns = data.columns.astype(str)
    return data


def load(dataset: str, root: str | Path, omics: tuple[str, ...] = ("mrna",)) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns the features (samples x features) and the encoded labels of one MLOmics dataset."""
    assert dataset in DATASETS, f"{dataset} not in {DATASETS}"
    folder = Path(root) / "Main_Dataset" / "Classification_datasets" / f"GS-{dataset}" / "Original"
    X = pd.concat([_load_omic(folder / f"{dataset}_{OMIC_FILES[o]}.csv") for o in omics], axis=1)
    labels = pd.read_csv(folder / f"{dataset}_label_num.csv").values.flatten()  # same row order as X
    return X, LabelEncoder().fit_transform(labels)


def reduce_features(X: pd.DataFrame, n_features: int) -> np.ndarray:
    """Feature agglomeration to n_features (0 or more than available: unchanged), as TabPFN-Wide.
    Like them, it is fitted on all samples: it does not use the labels, but it does see the test rows."""
    if n_features == 0 or X.shape[1] <= n_features:
        return X.values.astype(np.float32)
    return FeatureAgglomeration(n_clusters=n_features).fit_transform(X).astype(np.float32)


def folds(y: np.ndarray):
    """The (train_idx, test_idx) pairs TabPFN-Wide uses: stratified 5-fold, one repeat, seed 42."""
    return list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(np.zeros(len(y)), y))


def published_results(tabpfn_wide_root: str | Path, model: str) -> pd.DataFrame:
    """Per-fold results published by TabPFN-Wide for one model directory, e.g. "wide-v2-5k",
    "v2", "tabicl", "random_forest", "xgboost", "realmlp"."""
    path = Path(tabpfn_wide_root) / "analysis_results" / model / "multiomics_feature_reduction_results.csv"
    return pd.read_csv(path)


def check_folds_match(published: pd.DataFrame, dataset: str, n_features: int, y: np.ndarray) -> bool:
    """True if our test folds have exactly the same labels, in the same order, as the ones in the
    published results (their 'ground_truth' column) for this dataset and feature count."""
    rows = published[(published.dataset_name == dataset) & (published.n_features == n_features)].sort_values("fold")
    ours = folds(y)
    if len(rows) != len(ours):
        return False
    for (_, test_idx), (_, row) in zip(ours, rows.iterrows()):
        theirs = np.array(str(row.ground_truth).split(), dtype=int)
        if not np.array_equal(theirs, y[test_idx]):
            return False
    return True
