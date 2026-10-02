import numpy as np
import pandas as pd
import pytest

from tfmplayground.benchmarks import mlomics
from tfmplayground.benchmarks.metrics import accuracy, roc_auc
from tfmplayground.benchmarks.runner import compare, run_mlomics


def _fake_mlomics(root, dataset="BRCA", n_samples=60, n_genes=40, n_classes=3, seed=0):
    """Writes a tiny dataset with the MLOmics layout: features x samples CSVs and a label file."""
    rng = np.random.default_rng(seed)
    folder = root / "Main_Dataset" / "Classification_datasets" / f"GS-{dataset}" / "Original"
    folder.mkdir(parents=True)
    y = np.arange(n_samples) % n_classes
    values = rng.normal(size=(n_genes, n_samples)) + y[None, :] * (np.arange(n_genes) < 5)[:, None]
    genes = [f"g{i}" for i in range(n_genes)]
    df = pd.DataFrame(values, index=genes, columns=[f"s{j}" for j in range(n_samples)])
    df.iloc[3] = 0.0  # an all-zero gene, to be dropped
    df.iloc[7, 2] = np.nan  # a missing value, to be filled with the gene mean
    df = pd.concat([df, df.iloc[[10]]])  # a duplicated gene, to be dropped
    df.to_csv(folder / f"{dataset}_mRNA.csv")
    pd.DataFrame({"Label": y}).to_csv(folder / f"{dataset}_label_num.csv", index=False)
    return y


def test_load_cleans_like_tabpfn_wide(tmp_path):
    y_written = _fake_mlomics(tmp_path)
    X, y = mlomics.load("BRCA", tmp_path)
    assert X.shape == (60, 39)  # 40 genes - 1 all-zero; the duplicate is dropped
    assert "g3" not in X.columns and not X.isna().any().any()
    assert np.array_equal(y, y_written)


def test_reduce_features(tmp_path):
    _fake_mlomics(tmp_path)
    X, _ = mlomics.load("BRCA", tmp_path)
    assert mlomics.reduce_features(X, 0).shape == (60, 39)
    assert mlomics.reduce_features(X, 100).shape == (60, 39)  # more than available: unchanged
    assert mlomics.reduce_features(X, 10).shape == (60, 10)


def test_folds_are_deterministic_and_stratified():
    y = np.arange(100) % 4
    a, b = mlomics.folds(y), mlomics.folds(y)
    assert len(a) == 5
    for (tr1, te1), (tr2, te2) in zip(a, b):
        assert np.array_equal(te1, te2) and np.array_equal(tr1, tr2)
        assert np.bincount(y[te1]).tolist() == [5, 5, 5, 5]


def test_check_folds_match_detects_differences():
    y = np.arange(50) % 2
    rows = [
        {"dataset_name": "BRCA", "n_features": 30, "fold": i, "ground_truth": " ".join(map(str, y[te]))}
        for i, (_, te) in enumerate(mlomics.folds(y))
    ]
    published = pd.DataFrame(rows)
    assert mlomics.check_folds_match(published, "BRCA", 30, y)
    published.loc[0, "ground_truth"] = " ".join(["1"] * 10)
    assert not mlomics.check_folds_match(published, "BRCA", 30, y)


def test_metrics_match_tabpfn_wide_definition():
    y = np.array([0, 1, 1, 0])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.3, 0.7]])
    assert roc_auc(y, proba) == pytest.approx(0.75)
    assert accuracy(y, proba) == pytest.approx(0.75)
    y3 = np.array([0, 1, 2, 2])
    proba3 = np.eye(3)[y3] * 0.8 + 0.2 / 3
    assert roc_auc(y3, proba3) == pytest.approx(1.0)


def test_run_mlomics_with_random_forest_resumes(tmp_path):
    _fake_mlomics(tmp_path)
    out = tmp_path / "results.csv"
    res = run_mlomics(["random_forest"], tmp_path, out, datasets=("BRCA",), n_features=(0, 10), device="cpu")
    assert len(res) == 10  # 2 feature counts x 5 folds
    assert res.roc_auc.between(0, 1).all()
    again = run_mlomics(["random_forest"], tmp_path, out, datasets=("BRCA",), n_features=(0, 10), device="cpu")
    assert len(again) == 10  # nothing recomputed

    published = {"rf": res.assign(omic="mrna", checkpoint="x")}
    table = compare(res, published)
    assert set(table.checkpoint) == {"random_forest", "[TabPFN-Wide] rf"}
    assert (table["count"] == 5).all()
