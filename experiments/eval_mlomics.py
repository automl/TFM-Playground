"""Evaluates models on the MLOmics TCGA benchmark with the TabPFN-Wide protocol (stratified 5-fold,
seed 42, same preprocessing) and appends per-fold results to a CSV. Then prints a comparison with
the per-fold results TabPFN-Wide published.

Example, from the repo root:
    python experiments/eval_mlomics.py --gpu 5 --models random_forest workdir/graph_scm_base/epoch_40.pth
Use epoch snapshots rather than latest_checkpoint.pth: the name is what identifies the model in the CSV.
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from tfmplayground.benchmarks import mlomics
from tfmplayground.benchmarks.runner import compare, run_mlomics

REPO = Path(__file__).resolve().parents[1]
PUBLISHED = ("wide-v2-5k", "v2", "tabicl", "random_forest", "xgboost", "realmlp")

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", required=True, help="random_forest, logreg or checkpoint paths")
parser.add_argument("--datasets", nargs="+", default=list(mlomics.DATASETS))
parser.add_argument("--omics", nargs="+", default=["mrna"], help="mrna, cnv, methylation, mirna")
parser.add_argument("--n-features", nargs="+", type=int, default=[0], help="0 = all features")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--output", default="workdir/eval/mlomics.csv")
parser.add_argument("--mlomics-root", default=str(REPO.parent / "Cancer-Multi-Omics-Benchmark"))
parser.add_argument("--tabpfn-wide-root", default=str(REPO.parent / "TabPFN-Wide"))
args = parser.parse_args()

os.chdir(REPO)
results = run_mlomics(
    models=args.models,
    mlomics_root=args.mlomics_root,
    output=args.output,
    datasets=tuple(args.datasets),
    omics=tuple(args.omics),
    n_features=tuple(args.n_features),
    device=f"cuda:{args.gpu}",
    tabpfn_wide_root=args.tabpfn_wide_root if Path(args.tabpfn_wide_root).exists() else None,
)
if Path(args.tabpfn_wide_root).exists() and tuple(args.omics) == ("mrna",):
    published = {m: mlomics.published_results(args.tabpfn_wide_root, m) for m in PUBLISHED}
    table = compare(results[results.omic == "mrna"], published)
    pd.set_option("display.width", 200)
    for (dataset, n), group in table.groupby(["dataset_name", "n_features"]):
        print(f"\n{dataset} | {n} features (ROC AUC, mean ± std over 5 folds)")
        for _, r in group.sort_values("mean", ascending=False).iterrows():
            print(f"  {r.checkpoint:>45}: {r['mean']:.3f} ± {r['std']:.3f}")
