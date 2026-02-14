"""
Real Data Prior - Cache Builder (Phase 1)

Purpose:
Build a local on-disk cache of cleaned and preprocessed OpenML datasets.
Each dataset is downloaded once, cleaned, encoded, imputed, and stored as a
compressed .npz file along with metadata for later use.

This cache is used by:
- Phase 2: Pool selection (train/eval splits, duplicate detection)
- Phase 3: Episode generation (sampling in context learning episodes)

What this script does:
- Downloads OpenML datasets by ID
- Cleans and filters unusable columns
- Encodes categorical features
- Imputes missing values
- Computes statistical fingerprints for duplicate detection
- Stores dataset + metadata locally for fast reuse

Example usage:
    python -m tfmplayground.priors.real_data.cache_builder \
        --dataset-csv path/to/datasets.csv \
        --cache-dir data/cache

Optional arguments:
    --max-datasets N
    --max-rows N
    --max-features N
    --no-skip-existing

Output:
    cache_dir/
        ├── datasets/ (compressed .npz files)
        ├── metadata.json
        └── failed_datasets.json (if any)
"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer

@dataclass
class DatasetMetadata:
    """metadata to store for a cached dataset"""
    dataset_id: str
    dataset_name: str
    source_type: str  # "openml" or TODO:"kaggle" later on
    cache_path: str
    n_rows: int
    n_features: int # excluding the target
    column_names: list[str]
    column_is_categorical: list[bool]  # per column w/ target
    column_unique_counts: list[int]    # per column w/ target (used for automatic task detection for the mixed case)
    original_target_idx: int           # index of the original target in the flattened matrix
    column_means: list[float]          # per-feature stats (target excluded), used for duplicate detection
    column_vars: list[float]
    column_skews: list[float]
    column_kurtoses: list[float]
    y_mean: float # after encoding
    y_var: float
    y_unique_count: int
    intended_task_type: str   # default task type of original target (OpenML metadata or fallback heuristic)
    frac_missing: float  # fraction of missing values in the dataset
    data_hash: str = ""  # for detecting identical copies under different ids using SHA1 hex digest
    task_id: Optional[int] = None  # OpenML task ID if available


def impute_missing_values(data: np.ndarray, is_categorical: list[bool]) -> np.ndarray:
    """impute missing values on all columns
    encoding already happened upstream in load_openml_dataset
    """
    data = data.copy()

    # impute categorical columns with mode, numeric columns with mean
    cat_idx = [i for i, is_cat in enumerate(is_categorical) if is_cat]
    num_idx = [i for i, is_cat in enumerate(is_categorical) if not is_cat]

    if cat_idx:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        data[:, cat_idx] = cat_imputer.fit_transform(data[:, cat_idx])

    if num_idx:
        num_imputer = SimpleImputer(strategy='mean')
        data[:, num_idx] = num_imputer.fit_transform(data[:, num_idx])

    return data


def compute_column_statistics(data: np.ndarray) -> dict:
    """compute per-feature statistics (mean, variance, skewness, kurtosis)
    for duplicate detection and dataset fingerprinting."""
    means = np.nanmean(data, axis=0).tolist()
    variances = np.nanvar(data, axis=0).tolist()

    # skewness and kurtosis calculation
    skewness = []
    kurtosis = []
    for i in range(data.shape[1]):
        col = data[:, i]
        col = col[~np.isnan(col)]
        if len(col) > 2:
            skewness.append(float(stats.skew(col)))
            kurtosis.append(float(stats.kurtosis(col)))
        else:
            skewness.append(0.0)
            kurtosis.append(0.0)

    # replace inf/nan with 0 to keep calculation sane
    means = [0.0 if not np.isfinite(x) else x for x in means]
    variances = [0.0 if not np.isfinite(x) else x for x in variances]
    skewness = [0.0 if not np.isfinite(x) else x for x in skewness]
    kurtosis = [0.0 if not np.isfinite(x) else x for x in kurtosis]

    return {
        "column_means": means,
        "column_vars": variances,
        "column_skewness": skewness,
        "column_kurtosis": kurtosis,
    }


def clean_dataframe(df: pd.DataFrame, target_name: str,
                    cardinality_threshold: float = 0.05) -> Optional[pd.DataFrame]:
    """
    cleaning the raw DataFrame before caching.
    - drop image path like string columns
    - drop constant columns
    - infer which remaining columns are numeric
    - detect categorical columns via normalised cardinality
    - keep only numeric or categorical columns, always keep target
    Returns cleaned DataFrame, or None if the dataset becomes unusable.
    """
    if target_name not in df.columns:
        return None

    columns_to_drop: list[str] = []

    # drop image path like string columns
    obj_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    for col in obj_cols:
        if (df[col].astype(str)
                .str.contains(r"\.(?:jpg|jpeg|png)$", case=False, regex=True, na=False)
                .any()):
            columns_to_drop.append(col)

    # drop constant columns (nunique <= 1)
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            columns_to_drop.append(col)

    columns_to_drop = sorted(set(columns_to_drop))

    # safety guard to not drop target
    if target_name in columns_to_drop:
        return None

    # apply the drop
    df = df.drop(columns=columns_to_drop, errors="ignore")

    # infer numeric columns
    is_numeric = pd.Series(False, index=df.columns)

    for col in df.columns:
        # mark the numeric ones
        if pd.api.types.is_numeric_dtype(df[col]):
            is_numeric[col] = True
        else:
            # for numerics that are stored as strings
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
                is_numeric[col] = True
            except (ValueError, TypeError):
                pass

    # detect categorical ones using normalised cardinality
    # use valid count as denominator so sparse columns don't look like low cardinality
    n_valid = df.count()
    col_norm_card = df.nunique() / n_valid
    is_categorical_mask = (col_norm_card < cardinality_threshold) & ~is_numeric

    # keep only numeric or categorical columns, always keep target
    cols_mask = is_numeric | is_categorical_mask
    cols_mask[target_name] = True
    df = df.loc[:, cols_mask]

    # final guards for unusable datasets
    if df.shape[1] < 2 or df.shape[0] < 2:
        return None

    return df


def load_openml_dataset(dataset_id: int):
    """
    load and preprocess an OpenML dataset.
    - downloads dataset and target.
    - normalizes common missing markers.
    - cleans unusable columns.
    - infers task type (classification/regression).
    - encodes categorical columns.

    Returns:
        data (np.ndarray): numeric matrix with target as last column.
        column_names (list[str])
        is_categorical (list[bool])
        column_unique_counts (list[int])
        original_target_idx (int)
        dataset_name (str)
        task_type (str)

    Returns None if the dataset is unusable.
    """

    import openml

    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=True
    )

    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format="dataframe"
    )

    # normalize common missing markers with nan's
    X = X.replace(["?"], np.nan)
    X = X.replace([r"^-+$"], np.nan, regex=True)
    X = X.replace("", np.nan)

    if y is not None:
        y = y.replace(["?"], np.nan)
        y = y.replace([r"^-+$"], np.nan, regex=True)
        y = y.replace("", np.nan)
    else:
        return None  # cannot use datasets without a target

    # extract the target name default or 'target'
    target_name = dataset.default_target_attribute or "target"

    # combine into one df for cleaning and mixed sampling option
    df = X.copy()
    df[target_name] = y

    # drop image paths, constant cols, non-numeric and non-categorical columns
    df = clean_dataframe(df, target_name=target_name)

    # safety check for if all cleaned
    if df is None:
        return None

    # determine task type from OpenML metadata
    target_is_nominal = False
    if dataset.features is not None:
        for feat in dataset.features.values():
            if feat.name == target_name:
                target_is_nominal = (feat.data_type == "nominal")
                break

    # if target is nominal, task is classification, else regression
    if target_is_nominal:
        task_type = "classification"
    else:
        n_unique = df[target_name].nunique()
        # if target has 10 or less unique values, it is classification, else regression
        # NOTE: this is a heuristic and may not be correct for all datasets
        task_type = "classification" if n_unique <= 10 else "regression"

    # collect all object style aka categorical columns
    obj_cols = df.select_dtypes(include=["object", "string", "category"]).columns

    # encode all object columns into integer codes '-1' is used for missing values
    for col in obj_cols:
        df[col] = pd.Categorical(df[col])
        df[col] = df[col].cat.codes
        # replace -1 with nan's to keep the property
        df[col] = df[col].replace(-1, np.nan)

    # filter out small or degenerate datasets
    if len(df) < 10 or len(df.columns) < 2 or df[target_name].nunique() < 2:
        return None

    # make sure target is the last column cuz the entire pipeline logic depends on that
    feature_cols = [c for c in df.columns if c != target_name]
    ordered_cols = feature_cols + [target_name]
    df = df[ordered_cols]
    column_names = list(ordered_cols)

    # rebuild is_categorical cuz we did preprocessing and it doesn't necessarily match the og dataset now
    is_categorical = []
    column_unique_counts = []
    for col in ordered_cols:
        if col == target_name:
            is_categorical.append(task_type == "classification")
        else:
            is_categorical.append(col in obj_cols)
        column_unique_counts.append(int(df[col].nunique()))
    original_target_idx = len(ordered_cols) - 1  # target is last

    # convert df to numpy array
    data = df.values
    return data, column_names, is_categorical, column_unique_counts, original_target_idx, dataset.name, task_type


def parse_dataset_csv(csv_path: str) -> list[dict]:
    """parse a CSV file to extract dataset IDs and optional task IDs"""

    df = pd.read_csv(csv_path, sep=";")
    datasets = []

    # extract the relevant columns
    id_columns = ['openml_id']
    task_id_columns = ['openml_tid']

    if "openml_id" not in df.columns:
        raise ValueError(f"Could not find OpenML ID column in {csv_path}. Expected: {id_columns}")

    for _, row in df.iterrows():
        did = row["openml_id"]
        # tid is not present in all of them
        tid = row["openml_tid"] if "openml_tid" in df.columns else None

        if pd.notna(did):
            entry = {
                "dataset_id": int(did),
                #TODO: currently its all openml so this is hardcoded
                "source": "openml",
            }
            if pd.notna(tid):
                entry["task_id"] = int(float(tid))
            datasets.append(entry)
    return datasets


def save_dataset_cache(data: np.ndarray, column_names: list[str],
                       column_is_categorical: list[bool], column_unique_counts: list[int],
                       original_target_idx: int, task_type: str, cache_path: str):
    """save preprocessed dataset to .npz file.
    store everything needed to reconstruct and sample from the dataset later
    without re-downloading or re-processing
    """

    np.savez_compressed(
        cache_path,
        data=data,
        column_names=np.array(column_names, dtype=object),
        column_is_categorical=np.array(column_is_categorical),
        column_unique_counts=np.array(column_unique_counts),
        original_target_idx=np.array(original_target_idx),
        intended_task_type=np.array(task_type, dtype=object)
    )


def build_cache(
    dataset_csv: str,
    cache_dir: str = "data/cache",
    max_datasets: Optional[int] = None,
    max_rows: Optional[int] = None,
    max_features: Optional[int] = None,
    skip_existing: bool = True,
):
    """build cache from a CSV of openml dataset IDs."""
    cache_dir = Path(cache_dir)
    datasets_dir = cache_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # load existing metadata if exists
    metadata_path = cache_dir / "metadata.json"
    # this is to keep track of the datasets that have already been processed 
    # so we can do updates rather than creating the cache from scratch for each new dataset
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            existing_metadata = json.load(f)
            cached_ids = {m["dataset_id"] for m in existing_metadata.get("datasets", [])}
    else:
        existing_metadata = {"datasets": [], "created": datetime.now().isoformat()}
        cached_ids = set()

    # parse the dataset CSV
    openml_datasets = parse_dataset_csv(dataset_csv)
    print(f"Found {len(openml_datasets)} datasets in {dataset_csv}")
    print(f"Processing {len(openml_datasets)} OpenML datasets")

    failed = []
    processed = 0

    for i, info in enumerate(openml_datasets):
        # stop if we have processed enough datasets
        if max_datasets is not None and processed >= max_datasets:
            break

        dataset_id = info["dataset_id"]

        # skip if already cached based on the metadata
        if skip_existing and str(dataset_id) in cached_ids:
            print(f"[{i+1}/{len(openml_datasets)}] Skipping {dataset_id} (already cached)")
            continue

        print(f"[{i+1}/{len(openml_datasets)}] Processing dataset {dataset_id}...", end=" ", flush=True)
        start_time = time.time()

        try:
            # load from OpenML by dataset ID
            loaded = load_openml_dataset(dataset_id)
            if loaded is None:
                print("SKIP (dropped target, empty, or no target)")
                continue
            raw_data, column_names, is_categorical, column_unique_counts, original_target_idx, name, task_type = loaded

            n_rows, n_cols = raw_data.shape
            n_features = n_cols - 1  # excluding target

            # apply size filters if provided
            if max_rows and n_rows > max_rows:
                print(f"SKIP (too many rows: {n_rows} > {max_rows})")
                continue
            if max_features and n_features > max_features:
                print(f"SKIP (too many features: {n_features} > {max_features})")
                continue

            # compute missing fraction before preprocessing
            total_cells = raw_data.size
            missing_cells = np.sum(pd.isna(raw_data))
            frac_missing = missing_cells / total_cells if total_cells > 0 else 0.0

            data_proc = impute_missing_values(raw_data, is_categorical)
            data_proc = data_proc.astype(np.float32)

            # fingerprint of the processed matrix to detect exact duplicates across dataset ids
            data_hash = hashlib.sha1(data_proc.tobytes()).hexdigest()

            # compute column statistics for features only, no target
            feature_data = np.delete(data_proc, original_target_idx, axis=1)
            col_stats = compute_column_statistics(feature_data)

            # compute original target statistics
            y_col = data_proc[:, original_target_idx]

            # save to cache
            cache_path = datasets_dir / f"openml_{dataset_id}.npz"
            save_dataset_cache(
                data_proc, column_names, is_categorical, column_unique_counts,
                original_target_idx, task_type, str(cache_path)
            )

            # create metadata
            metadata = DatasetMetadata(
                dataset_id=str(dataset_id),
                source_type="openml",
                cache_path=f"datasets/openml_{dataset_id}.npz",
                n_rows=n_rows,
                n_features=n_features,
                column_names=column_names,
                column_is_categorical=is_categorical,
                column_unique_counts=column_unique_counts,
                original_target_idx=original_target_idx,
                column_means=col_stats["column_means"],
                column_vars=col_stats["column_vars"],
                column_skews=col_stats["column_skewness"],
                column_kurtoses=col_stats["column_kurtosis"],
                y_mean=float(np.mean(y_col)),
                y_var=float(np.var(y_col)),
                y_unique_count=int(len(np.unique(y_col[~np.isnan(y_col)]))),
                intended_task_type=task_type,
                frac_missing=float(frac_missing),
                dataset_name=name or f"openml_{dataset_id}",
                data_hash=data_hash,
                task_id=info.get("task_id"),
            )

            # add to metadata
            existing_metadata["datasets"].append(asdict(metadata))
            cached_ids.add(str(dataset_id))

            elapsed = time.time() - start_time
            print(f"OK ({n_rows} x {n_features}, {task_type}, {elapsed:.1f}s)")            
            processed += 1

            # save metadata periodically
            if processed % 10 == 0:
                with open(metadata_path, "w") as f:
                    json.dump(existing_metadata, f, indent=2)

        except Exception as e:
            print(f"FAILED: {e}")
            failed.append({"dataset_id": dataset_id, "error": str(e)})

    # final save of everything
    existing_metadata["last_updated"] = datetime.now().isoformat()
    with open(metadata_path, "w") as f:
        json.dump(existing_metadata, f, indent=2)

    # save failed datasets as a log
    if failed:
        failed_path = cache_dir / "failed_datasets.json"
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)

    # print summary
    print("\n" + "=" * 50)
    print(f"CACHE BUILD COMPLETE")
    print("=" * 50)
    print(f"Successfully cached: {processed} datasets")
    print(f"Failed: {len(failed)} datasets")
    print(f"Total in cache: {len(existing_metadata['datasets'])} datasets")
    print(f"Cache directory: {cache_dir}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Build cache for Real Data Prior")
    parser.add_argument("--dataset-csv", type=str, default="tfmplayground/priors/real_data/openml.csv",
                        help="Path to CSV with dataset IDs")
    parser.add_argument("--cache-dir", type=str, default="tfmplayground/priors/real_data/data/cache",
                        help="Directory to save cache")
    parser.add_argument("--max-datasets", type=int, default=None,
                        help="Stop after caching this many datasets")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Skip datasets with more rows than this")
    parser.add_argument("--max-features", type=int, default=None,
                        help="Skip datasets with more features than this")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Re-process datasets that are already cached")
    args = parser.parse_args()

    build_cache(
        dataset_csv=args.dataset_csv,
        cache_dir=args.cache_dir,
        max_datasets=args.max_datasets,
        max_rows=args.max_rows,
        max_features=args.max_features,
        skip_existing=not args.no_skip_existing,
    )

if __name__ == "__main__":
    main()