"""
Real Data Prior - Pool Selector (Phase 2)

Purpose:
Select evaluation and training dataset pools from the Phase 1 cache.
Evaluation pools are defined by named suites. Training pools are formed from
the remaining cached datasets after excluding evaluation datasets and applying
optional filters and deduplication.

This phase is used by:
- Benchmarking and reporting on fixed evaluation suites
- Phase 3: Episode generation, where training datasets are sampled for in context episodes

What this script does:
- Loads Phase 1 metadata.json and builds fast lookup tables
- Resolves requested evaluation suites into dataset IDs plus split specifications
  - If a suite provides dataset IDs, creates deterministic random splits using a stable seed
  - If a suite provides OpenML task IDs, uses the official OpenML task split definition
- Builds the training candidate pool by removing evaluation datasets from the cache
- Optionally filters training datasets by size and missingness
- Optionally filters training datasets by task type (classification, regression, mixed)
- Optionally removes leakage by dropping training datasets that are duplicates of any eval dataset
- Optionally deduplicates within the training pool
- Writes output pool files and a report of all decisions

Duplicate detection:
Uses a multi signal fingerprinting approach based on cached statistics:
- Exact duplicate if processed content hash matches (data_hash)
- Additional similarity signals including:
  - same row count, ignoring round numbers
  - same feature count
  - similar target mean and variance
  - per feature distribution similarity using KD tree matching over column stats
Duplicates are reported with reasons and can be removed from training to avoid leakage.

Example usage:
    python -m tfmplayground.priors.real_data.pool_selector \
        --cache-dir tfmplayground/priors/real_data/data/cache \
        --suites-config tfmplayground/priors/real_data/data/suites_config.json \
        --suites cc18,ctr23 \
        --output-dir tfmplayground/priors/real_data/data/pools

Optional arguments:
    --strategy {exact_id_only,stats_fingerprint}
    --threshold FLOAT
    --no-train-dup
    --min-rows N
    --max-rows N
    --min-features N
    --max-features N
    --max-missing-frac FLOAT
    --seed N

Output:
    output_dir/
        ├── eval_pool_<suite_names>.json (only if suites were provided)
        ├── train_pool_all.txt
        ├── train_pool_classification.txt
        ├── train_pool_regression.txt
        └── report_<suite_names>.json
"""
import argparse
import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

from scipy.spatial import KDTree


@dataclass
class PoolReport:
    """tracks the decisions made during pool selection."""
    timestamp: str
    requested_suites: list[str]
    parameters: dict
    
    # data requested by suites but wasn't found in cache
    eval_missing_task_ids: dict[str, list[int]]
    eval_missing_dataset_ids: dict[str, list[str]]
    
    # training filtering
    train_removed_by_size: list[str] # row/column size filter
    train_removed_as_duplicates: list[dict]  # leaks between train and eval
    train_removed_by_dedupe: list[dict]      # training data dupes
    
    # final counts
    eval_count: int
    train_count_all: int
    train_count_classification: int
    train_count_regression: int


def load_metadata(cache_dir: Path) -> tuple[dict, dict[str, dict], dict[int, str]]:
    """load the metadata and build lookup structures.
    - full_metadata: the raw metadata
    - meta_by_id: dict[dataset_id_str, metadata_entry]
    - dataset_id_by_task_id: dict[task_id_int, dataset_id_str] (for suite mapping)
    """
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    
    with open(metadata_path) as f:
        full_metadata = json.load(f)
    
    # create lookups for fast access during pool selection
    meta_by_id = {}
    # map OpenML task_id -> dataset_id so suites can specify eval tasks
    # and we can resolve them to dataset IDs (which then get excluded from train)
    dataset_id_by_task_id = {}
    
    for entry in full_metadata.get("datasets", []):
        did = str(entry["dataset_id"])
        meta_by_id[did] = entry
        
        # map task_id -> dataset_id
        if entry.get("task_id") is not None:
            dataset_id_by_task_id[int(entry["task_id"])] = did
    
    return full_metadata, meta_by_id, dataset_id_by_task_id


def load_suites_config(config_path: Path) -> dict:
    """load suites configuration file."""
    with open(config_path) as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Suites config must be a JSON object")

    return config


def stable_int_seed(*parts: str, base_seed: int = 0) -> int:
    """generate a deterministic seed from string parts."""
    # this function helps it so that evaluation datasets that don't have a predefined split
    # get a random split that is the *same* every time the code is run based on the base seed
    # this is done by using a hash of the dataset id and the suite name.
    s = "|".join([str(base_seed)] + [str(p) for p in parts])
    h = hashlib.sha1(s.encode("utf8")).hexdigest()
    return int(h[:8], 16)


def resolve_eval_specs(
    requested_suites: list[str],
    suites_config: dict,
    meta_by_id: dict[str, dict],
    dataset_id_by_task_id: dict[int, str],
    report: PoolReport,
    base_seed: int,
    train_ratio: float = 0.7,  # train fraction for deterministic random split for eval sets
) -> dict[str, dict]:
    """
    for future benchmarking purposes
    resolve requested suites into eval dataset IDs with split specifications.
    """

    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be between 0 and 1. Got {train_ratio}")

    eval_specs: dict[str, dict] = {}

    for suite_name in requested_suites:
        if suite_name not in suites_config:
            raise ValueError(
                f"Suite '{suite_name}' not found in config. Available: {list(suites_config.keys())}"
            )

        suite = suites_config[suite_name]

        # apply deterministic random split conditions
        direct_ids = [str(did) for did in suite.get("eval_dataset_ids", [])]
        for did in direct_ids:
            # if not in cache, skip
            if did not in meta_by_id:
                report.eval_missing_dataset_ids.setdefault(suite_name, []).append(did)
                continue

            # output a deterministic seed to use for evaluation
            seed = stable_int_seed("suite", suite_name, "did", did, base_seed=base_seed)
            eval_specs[did] = {
                "split_type": "random",
                "seed": seed,
                "train_frac": float(train_ratio),
            }

        # output the official OpenML split conditions
        # task based splits intentionally overwrite random split outputs
        # since the same dataset can be present in multiple suites once as a dataset id
        # and once as a task id. in this case we want the task based split to take priority.
        missing_tasks: list[int] = []

        # some suites may have only tid
        for tid in suite.get("eval_task_ids", []):
            tid = int(tid)

            if tid in dataset_id_by_task_id:
                # match tid to did from cache
                did = dataset_id_by_task_id[tid]
                # skip if not found
                if did not in meta_by_id:
                    report.eval_missing_dataset_ids.setdefault(suite_name, []).append(did)
                    continue

                # specify tid along with did and fold conditions
                eval_specs[did] = {
                    "split_type": "openml_task",
                    "task_id": tid,
                    "fold": 0,
                    "repeat": 0,
                }
            else:
                missing_tasks.append(tid)

        # log missing tasks
        if missing_tasks:
            report.eval_missing_task_ids[suite_name] = missing_tasks
            print(f"Suite '{suite_name}': {len(missing_tasks)} task_ids not found in cache")

    return eval_specs


def build_column_kdtree(meta: dict) -> Optional[KDTree]:
    """build a KD-tree from per-column statistics for one dataset
    for checking similarity between datasets.
    each column becomes a 4D vector with pre-calculated: [mean, std, skew, kurtosis].

    Returns None if the metadata has no column statistics.
    """
    means = meta.get("column_means", [])
    variances = meta.get("column_vars", [])
    skews = meta.get("column_skews", [])
    kurtoses = meta.get("column_kurtoses", [])

    # check if stats are present
    if not means:
        return None

    # build a N×4 matrix: each row becomes one column's stats
    col_vectors = np.array([
        [m, np.sqrt(max(v, 0.0)), s, k]
        for m, v, s, k in zip(means, variances, skews, kurtoses)
    ], dtype=np.float64)

    col_vectors = np.nan_to_num(col_vectors, posinf=1.0, neginf=-1.0)

    # normalize per-row: each column's 4D vector is normalized within itself
    # this ensures consistent influence on similarity
    col_vectors = (col_vectors - col_vectors.mean(axis=1)[:, None]) / (
        col_vectors.std(axis=1)[:, None] + 1e-8 # avoid division by zero
    )

    return KDTree(col_vectors)


def is_duplicate(
    a_meta: dict,
    b_meta: dict,
    a_tree: Optional[KDTree] = None,
    b_tree: Optional[KDTree] = None,
    strategy: str = "stats_fingerprint",
    tree_threshold: float = 1e-3,
) -> tuple[bool, str]:
    """
    decide whether two cached datasets are duplicates.

    strategy:
    - if strategy == "exact_id_only": compare dataset_id only.
    - otherwise:
        1) immediate duplicate if processed content hash matches.
        2) accumulate similarity signals: equal (non-round) row count, equal feature count,
        similar target mean and variance, similar per-feature statistics via KD-tree matching
        3) classify as:
            - "high" confidence duplicate if >=3 signals
            - "low" confidence duplicate if >=2 signals or strong column similarity
            - not duplicate otherwise
    returns: (is_dup, reason)
    """

    # Fast path: only compare dataset IDs
    if strategy == "exact_id_only":
        return (
            (True, "same_dataset_id")
            if a_meta["dataset_id"] == b_meta["dataset_id"]
            else (False, "")
        )
    
    # Long path: use feature statistics for similarity checking
    # Check 1: SHA1 hash match/byte content is identical (certain duplication)
    a_hash = a_meta.get("data_hash", "")
    b_hash = b_meta.get("data_hash", "")
    if a_hash and b_hash and a_hash == b_hash:
        return True, "sha1_exact_match"

    # Check 2: Accumulation of different similarity metrics
    reasons = []
    high_feature_similarity = False

    # extract metadata from the 2 datasets
    n_rows_a = a_meta.get("n_rows", 0)
    n_rows_b = b_meta.get("n_rows", 0)
    n_features_a = a_meta.get("n_features", 0)
    n_features_b = b_meta.get("n_features", 0)

    # check the equality of row counts but ignore round numbers
    # 100 and 100 isn't as suspicious as 142 and 142
    if n_rows_a == n_rows_b and sum(c != "0" for c in str(n_rows_a)) > 1:
        reasons.append(f"same_n_rows_{n_rows_a}")

    # feature count match
    if n_features_a == n_features_b:
        reasons.append(f"same_n_features_{n_features_a}")

    # target statistics match both mean and variance are almost equal
    if (
        np.isclose(a_meta.get("y_mean", 0), b_meta.get("y_mean", 0))
        and np.isclose(a_meta.get("y_var", 0), b_meta.get("y_var", 0))
    ):
        reasons.append("similar_target_stats")

    # similarity of columns between the datasets using KD-tree
    if a_tree is not None and b_tree is not None:
        # returns a list of indices where a and b's columns are within the threshold (Euclidean Distance)
        pair_inds = a_tree.query_ball_tree(b_tree, tree_threshold)
        # count the number of similar ones for both datasets
        # the number of columns in 'a' that have at least one similar column in 'b'
        n_similar_a = sum(len(xs) > 0 for xs in pair_inds)
        # the number of columns in 'b' that have at least one similar column in 'a'
        n_similar_b = len(set(x for xs in pair_inds for x in xs))
        # take the overlap
        # prevents overcounting when one dataset matches many columns of the other
        n_similar = min(n_similar_a, n_similar_b)

        # feature overlap check but relative to the small feature set
        min_n_features = min(n_features_a, n_features_b)
        # threshold is min 5 or the number of features in the smaller dataset
        thresh = min(5, min_n_features)

        if n_similar >= thresh:
            reasons.append(f"{n_similar}_similar_columns")

            # dataset dimensionality ratio
            max_n_features = max(n_features_a, n_features_b)
            n_feature_ratio = max_n_features / max(min_n_features, 1)

            # detection fraction
            # fraction becomes smaller for larger feature sets
            # and higher for small feature sets cuz matching few columns in small sets are already sus
            detection_frac = min(1, 1 / 2 ** (np.log10(max(min_n_features, 1)) - 1))

            # if 'enough' features match and the sizes are not wildly different, it's a duplicate
            if n_similar > detection_frac * min_n_features and n_feature_ratio < 1.5:
                high_feature_similarity = True

    # 3 or more signals are strong enough to consider it a duplicate
    if len(reasons) > 2:
        return True, "high:" + ",".join(reasons)
    # 2 signals OR high feature similarity is duplicate
    elif len(reasons) > 1 or high_feature_similarity:
        return True, "low:" + ",".join(reasons)

    return False, ""


def filter_train_by_size(
    train_ids: set[str],
    meta_by_id: dict[str, dict],
    min_rows: Optional[int],
    max_rows: Optional[int],
    min_features: Optional[int],
    max_features: Optional[int],
    max_missing_frac: Optional[float],
) -> tuple[set[str], list[str]]:
    """apply size filters to training candidates."""
    filtered = set()
    removed = []
    
    for did in train_ids:
        # extract the metadata needed for filtering
        meta = meta_by_id[did]
        n_rows = meta.get("n_rows", 0)
        n_features = meta.get("n_features", 0)
        frac_missing = meta.get("frac_missing", 0.0)
        
        keep = True
        reason = ""
        
        # apply filters and add the corresponding reasons behind the decisions
        if min_rows and n_rows < min_rows:
            keep = False
            reason = f"rows={n_rows} < min_rows={min_rows}"
        elif max_rows and n_rows > max_rows:
            keep = False
            reason = f"rows={n_rows} > max_rows={max_rows}"
        elif min_features and n_features < min_features:
            keep = False
            reason = f"features={n_features} < min_features={min_features}"
        elif max_features and n_features > max_features:
            keep = False
            reason = f"features={n_features} > max_features={max_features}"
        elif max_missing_frac and frac_missing > max_missing_frac:
            keep = False
            reason = f"missing_frac={frac_missing:.3f} > max={max_missing_frac}"
        
        if keep:
            filtered.add(did)
        else:
            removed.append(f"{did}:{reason}")
    
    return filtered, removed


def filter_train_by_task_mode(
    train_ids: set[str],
    meta_by_id: dict[str, dict],
    task_mode: str,
) -> tuple[set[str], list[str]]:
    """filter training by task mode (classification/regression/mixed)."""
    if task_mode == "mixed":
        return train_ids, []
        
    filtered = set()
    removed = []
    
    for did in train_ids:
        meta = meta_by_id[did]
        if meta.get("intended_task_type") == task_mode:
            filtered.add(did)
        else:
            removed.append(did)
    
    return filtered, removed


def remove_duplicates_of_eval_in_train(
    train_ids: set[str],
    eval_ids: set[str],
    meta_by_id: dict[str, dict],
    strategy: str,
    threshold: float,
    kd_trees: Optional[dict[str, Optional[KDTree]]] = None,
) -> tuple[set[str], list[dict]]:
    """remove training datasets that are duplicates of any evaluation dataset"""
    clean_train = set()
    removed = []
    trees = kd_trees or {}

    eval_metadata = [(did, meta_by_id[did]) for did in eval_ids]

    for train_did in train_ids:
        train_meta = meta_by_id[train_did]
        is_dup = False
        dup_info = None

        # compare it against all evaluation datasets
        for eval_did, eval_meta in eval_metadata:
            dup, reason = is_duplicate(
                train_meta, eval_meta,
                a_tree=trees.get(train_did),
                b_tree=trees.get(eval_did),
                strategy=strategy,
                tree_threshold=threshold,
            )
            if dup:
                is_dup = True
                dup_info = {
                    "dataset_id": train_did,
                    "duplicate_of": eval_did,
                    "reason": reason,
                }
                break

        if is_dup:
            removed.append(dup_info)
        else:
            clean_train.add(train_did)

    return clean_train, removed


def remove_duplicates_in_train(
    train_ids: set[str],
    meta_by_id: dict[str, dict],
    strategy: str,
    threshold: float,
    kd_trees: Optional[dict[str, Optional[KDTree]]] = None,
) -> tuple[set[str], list[dict]]:
    """remove duplicates within training pool, keeping first occurrence."""
    train_list = sorted(train_ids)  # deterministic order
    kept = []
    removed = []
    # if they exist since they are optional
    trees = kd_trees or {}

    for did in train_list:
        meta = meta_by_id[did]
        is_dup = False

        # check if any of the kept datasets are duplicates of the current one
        for kept_did in kept:
            kept_meta = meta_by_id[kept_did]
            dup, reason = is_duplicate(
                meta, kept_meta,
                a_tree=trees.get(did),
                b_tree=trees.get(kept_did),
                strategy=strategy,
                tree_threshold=threshold,
            )
            if dup:
                is_dup = True
                removed.append({
                    "dataset_id": did,
                    "duplicate_of": kept_did,
                    "reason": reason,
                })
                break
        # keep if not dupe
        if not is_dup:
            kept.append(did)

    return set(kept), removed


def write_pool_file(path: Path, dataset_ids: set[str]):
    """write pool file with sorted dataset IDs, one per line."""
    with open(path, "w") as f:
        for did in sorted(dataset_ids, key=lambda x: int(x) if x.isdigit() else x):
            f.write(f"{did}\n")


def write_eval_pool_json(path: Path, eval_specs: dict[str, dict]):
    """write eval pool as JSON with split specifications."""
    payload = []
    for did, spec in eval_specs.items():
        payload.append({"dataset_id": did, **spec})
    # sort by did
    payload.sort(key=lambda x: int(x["dataset_id"]) if str(x["dataset_id"]).isdigit() else str(x["dataset_id"]))
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def select_pools(
    cache_dir: str,
    suites_config: str,
    suites: list[str],
    output_dir: str,
    dup_strategy: str = "stats_fingerprint",
    dup_threshold: float = 1e-3,
    remove_dupes_in_train: bool = True,
    min_rows: Optional[int] = None,
    max_rows: Optional[int] = None,
    min_features: Optional[int] = None,
    max_features: Optional[int] = None,
    max_missing_frac: Optional[float] = None,
    seed: int = 0,
):
    """main pool selection logic."""
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # initialize pool report
    report = PoolReport(
        timestamp=datetime.now().isoformat(),
        requested_suites=suites,
        parameters={
            "dup_strategy": dup_strategy,
            "dup_threshold": dup_threshold,
            "dedupe_train": remove_dupes_in_train,
            "min_rows": min_rows,
            "max_rows": max_rows,
            "min_features": min_features,
            "max_features": max_features,
            "max_missing_frac": max_missing_frac,
            "seed": seed,
        },
        eval_missing_task_ids={},
        eval_missing_dataset_ids={},
        train_removed_by_size=[],
        train_removed_as_duplicates=[],
        train_removed_by_dedupe=[],
        eval_count=0,
        train_count_all=0,
        train_count_classification=0,
        train_count_regression=0,
    )
    
    # Step 1: load the cached metadata
    print("Loading metadata...")
    full_meta, meta_by_id, dataset_id_by_task_id = load_metadata(cache_dir)
    all_ids = set(meta_by_id.keys())
    print(f"  Found {len(all_ids)} datasets in cache")
    
    # Step 2: load the suites configurations
    print("Loading suites config...")
    suites_config_data = load_suites_config(Path(suites_config))
    
    # Step 3: create the deterministic split conditions for the eval set
    eval_specs = {}
    if suites:
        print(f"Resolving eval specs for suites: {suites}...")
        eval_specs = resolve_eval_specs(
            suites, suites_config_data, meta_by_id, dataset_id_by_task_id, report,
            base_seed=seed,
            train_ratio=0.7,
        )
    
    # returns the dids for the ones that only have the task ids
    eval_ids = set(eval_specs.keys())
    print(f"  Eval pool: {len(eval_ids)} datasets")
    
    # Step 4: canditate training pool with only removing eval ids
    train_ids = all_ids - eval_ids
    print(f"  Initial train candidates: {len(train_ids)}")
    
    # Step 5: apply the size filters if any
    train_ids, removed_by_size = filter_train_by_size(
        train_ids, meta_by_id,
        min_rows, max_rows, min_features, max_features, max_missing_frac
    )
    report.train_removed_by_size = removed_by_size
    if removed_by_size:
        print(f"  Removed by size filters: {len(removed_by_size)}")
    
    # Step 6: pre-build KD-trees for all datasets (they get built once and are reused for all comparisons)
    kd_trees: dict[str, Optional[KDTree]] = {}
    if dup_strategy == "stats_fingerprint":
        print("  Building KD-trees for column matching...")
        for did in (eval_ids | train_ids):
            kd_trees[did] = build_column_kdtree(meta_by_id[did])

    # Step 7: remove duplicates of eval sets based on the duplication strategy
    if eval_ids:
        train_ids, removed_dups = remove_duplicates_of_eval_in_train(
            train_ids, eval_ids, meta_by_id, dup_strategy, dup_threshold,
            kd_trees=kd_trees,
        )
        report.train_removed_as_duplicates = removed_dups
        if removed_dups:
            print(f"  Removed as eval duplicates: {len(removed_dups)}")

    # Step 8: optional train-train deduplication
    if remove_dupes_in_train:
        train_ids, removed_train_dups = remove_duplicates_in_train(
            train_ids, meta_by_id, dup_strategy, dup_threshold,
            kd_trees=kd_trees,
        )
        report.train_removed_by_dedupe = removed_train_dups
        if removed_train_dups:
            print(f"  Removed by train-train dedupe: {len(removed_train_dups)}")

    # Step 9: split by task type and write all three pool files
    cls_ids, _ = filter_train_by_task_mode(train_ids, meta_by_id, "classification")
    reg_ids, _ = filter_train_by_task_mode(train_ids, meta_by_id, "regression")

    # final counts
    report.eval_count = len(eval_ids)
    report.train_count_all = len(train_ids)
    report.train_count_classification = len(cls_ids)
    report.train_count_regression = len(reg_ids)
    
    # Step 10: write output files
    if suites:
        joined_suites = "_".join(sorted(suites))
    else:
        joined_suites = "no_eval_suites"
    
    if eval_specs:
        eval_path = output_dir / f"eval_pool_{joined_suites}.json"
        write_eval_pool_json(eval_path, eval_specs)

    write_pool_file(output_dir / "train_pool_all.txt", train_ids)
    write_pool_file(output_dir / "train_pool_classification.txt", cls_ids)
    write_pool_file(output_dir / "train_pool_regression.txt", reg_ids)

    report_path = output_dir / f"report_{joined_suites}.json"
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    
    # print summary of the selection
    print("\n" + "=" * 50)
    print("POOL SELECTION COMPLETE")
    print("=" * 50)
    print(f"Suites: {', '.join(suites) if suites else 'none'}")
    print(f"Eval pool:       {len(eval_ids)} datasets")
    print(f"Train (all):     {len(train_ids)} datasets")
    print(f"Train (cls):     {len(cls_ids)} datasets")
    print(f"Train (reg):     {len(reg_ids)} datasets")
    print(f"Output directory: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select eval and training pools from cache")
    
    parser.add_argument("--cache-dir", type=str, default="tfmplayground/priors/real_data/data/cache",
                        help="Path to cache folder with metadata.json")
    parser.add_argument("--suites-config", type=str, default="tfmplayground/priors/real_data/suites_config.json",
                        help="Path to suites configuration JSON")
    parser.add_argument("--suites", type=str, default="",
                        help="Comma-separated list of suite names (e.g., 'cc18' or 'cc18, ctr23')")
    parser.add_argument("--output-dir", type=str, default="tfmplayground/priors/real_data/data/pools",
                        help="Directory to write pool files")
    parser.add_argument("--strategy", type=str,
                        choices=["exact_id_only", "stats_fingerprint"],
                        default="stats_fingerprint",
                        help="Duplicate detection strategy (default: stats_fingerprint)")
    parser.add_argument("--threshold", type=float, default=1e-3,
                        help="KD-tree distance threshold for column matching (lower = stricter)")
    parser.add_argument(
        "--no-train-dup",
        action="store_true",
        help="Disable deduplication within the training pool"
    )
    parser.add_argument("--min-rows", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--min-features", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--max-missing-frac", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    
    if args.suites.strip() == "":
        suites_list = []
    else:
        suites_list = [s.strip() for s in args.suites.split(",") if s.strip()]
    
    select_pools(
        cache_dir=args.cache_dir,
        suites_config=args.suites_config,
        suites=suites_list,
        output_dir=args.output_dir,
        dup_strategy=args.strategy,
        dup_threshold=args.threshold,
        remove_dupes_in_train=not args.no_train_dup,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        min_features=args.min_features,
        max_features=args.max_features,
        max_missing_frac=args.max_missing_frac,
        seed=args.seed,
    )
