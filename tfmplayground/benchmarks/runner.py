"""Runs models on a benchmark fold by fold and appends the results to a CSV, so that an interrupted
evaluation can be resumed and every fold can later be compared with published per-fold results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tfmplayground.benchmarks import mlomics
from tfmplayground.benchmarks.metrics import accuracy, roc_auc
from tfmplayground.benchmarks.models import make_model, predict_proba

COLUMNS = ["dataset_name", "omic", "checkpoint", "n_features", "fold", "accuracy", "roc_auc"]


def run_mlomics(
    models: list[str],
    mlomics_root: str | Path,
    output: str | Path,
    datasets: tuple[str, ...] = mlomics.DATASETS,
    omics: tuple[str, ...] = ("mrna",),
    n_features: tuple[int, ...] = (0,),
    device: str = "cuda",
    tabpfn_wide_root: str | Path | None = None,
) -> pd.DataFrame:
    """Evaluates every model on every dataset and feature count with the TabPFN-Wide protocol.
    If tabpfn_wide_root is given, first checks that our test folds are identical to theirs."""
    output = Path(output)
    results = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=COLUMNS)
    omic_name = "+".join(omics)
    published = None
    if tabpfn_wide_root is not None and omics == ("mrna",):
        published = mlomics.published_results(tabpfn_wide_root, "random_forest")

    for dataset in datasets:
        X_full, y = mlomics.load(dataset, mlomics_root, omics)
        for n in n_features:
            if n > X_full.shape[1]:
                print(f"{dataset}: skipping {n} features, only {X_full.shape[1]} available", flush=True)
                continue
            X = mlomics.reduce_features(X_full, n)
            folds = mlomics.folds(y)
            if published is not None:
                same = mlomics.check_folds_match(published, dataset, X.shape[1], y)
                print(f"{dataset} | {X.shape[1]} features | folds identical to TabPFN-Wide: {'yes' if same else 'NO'}", flush=True)
            for name in models:
                done = set(
                    results[
                        (results.dataset_name == dataset) & (results.omic == omic_name)
                        & (results.checkpoint == name) & (results.n_features == X.shape[1])
                    ].fold
                )
                for i, (train_idx, test_idx) in enumerate(folds):
                    if i in done:
                        continue
                    proba = predict_proba(make_model(name, device), X[train_idx], y[train_idx], X[test_idx])
                    row = {
                        "dataset_name": dataset, "omic": omic_name, "checkpoint": name, "n_features": X.shape[1],
                        "fold": i, "accuracy": accuracy(y[test_idx], proba), "roc_auc": roc_auc(y[test_idx], proba),
                    }
                    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                    print(f"{dataset} | {X.shape[1]} | {name} | fold {i} | AUROC {row['roc_auc']:.3f}", flush=True)
                output.parent.mkdir(parents=True, exist_ok=True)
                results.to_csv(output, index=False)
    return results


def compare(ours: pd.DataFrame, published: dict[str, pd.DataFrame], metric: str = "roc_auc") -> pd.DataFrame:
    """Mean and std over folds, per dataset and feature count, of our models and the published ones,
    side by side. Rows are matched on the actual number of features."""
    frames = [ours[["dataset_name", "checkpoint", "n_features", "fold", metric]]]
    for name, df in published.items():
        df = df[df.omic.str.lower() == "mrna"][["dataset_name", "n_features", "fold", metric]].copy()
        df["checkpoint"] = f"[TabPFN-Wide] {name}"
        frames.append(df)
    allres = pd.concat(frames, ignore_index=True)
    keys = set(map(tuple, ours[["dataset_name", "n_features"]].drop_duplicates().values))
    allres = allres[[k in keys for k in map(tuple, allres[["dataset_name", "n_features"]].values)]]
    stats = allres.groupby(["dataset_name", "n_features", "checkpoint"])[metric].agg(["mean", "std", "count"])
    return stats.reset_index()
