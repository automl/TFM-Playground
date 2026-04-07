# Real Data Priors

This folder contains the real-data prior pipeline used by `tfmplayground/priors/main.py` when `--lib real`.

## Overview

The pipeline has 3 phases:

1. Build cache of processed datasets from OpenML or Kaggle (`.npz` + metadata)
2. Select train/eval pools from that cache
3. Sample episodes from selected pools

## Files

- `build_dataset_cache.py`: downloads/processes datasets (OpenML or Kaggle) and writes cache
- `select_pools.py`: builds `train_pool_all.txt`, `train_pool_classification.txt`, `train_pool_regression.txt` (+ optional eval pool JSON)
- `episode_generator.py`: loads pools and emits episodes for classification/regression
- `suites_config.json`: eval suite definitions used by pool selection
- `openml.csv` / `kaggle.csv`: dataset lists for each source

## Phase 1: Build Cache

```bash
python -m tfmplayground.priors.real_data.build_dataset_cache \
  --dataset-csv tfmplayground/priors/real_data/openml.csv \
  --cache-dir tfmplayground/priors/real_data/data/cache
```

### Kaggle datasets

```bash
python -m tfmplayground.priors.real_data.build_dataset_cache \
  --dataset-csv tfmplayground/priors/real_data/kaggle.csv \
  --source kaggle \
  --cache-dir tfmplayground/priors/real_data/data/cache
```

Requires Kaggle API credentials:
```bash
EXPORT KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxx
```

The `kaggle.csv` format is: `kaggle_slug`

Both OpenML and Kaggle datasets share the same cache directory and metadata,
so pool selection and episode generation work transparently across sources.

Optional controls:
- `--source {openml,kaggle}` (default: openml)
- `--max-datasets`
- `--max-rows`
- `--max-features`
- `--no-skip-existing`

## Phase 2: Select Pools

```bash
python -m tfmplayground.priors.real_data.select_pools \
  --cache-dir tfmplayground/priors/real_data/data/cache \
  --suites-config tfmplayground/priors/real_data/suites_config.json \
  --suites cc18 \
  --output-dir tfmplayground/priors/real_data/data/pools
```

This writes:
- `train_pool_all.txt`
- `train_pool_classification.txt`
- `train_pool_regression.txt`
- `report_<...>.json`
- `eval_pool_<...>.json` (if suites are provided)

## Phase 3: Generate Real Prior Data

Run through `tfmplayground.priors` with `--lib real`.

### Classification, original-target only

```bash
python -m tfmplayground.priors \
  --lib real \
  --prior_type mlp \
  --max_classes 10 \
  --cache_dir tfmplayground/priors/real_data/data/cache \
  --train_pool tfmplayground/priors/real_data/data/pools/train_pool_classification.txt \
  --mode only
```

### Regression, original-target only

```bash
python -m tfmplayground.priors \
  --lib real \
  --prior_type mlp \
  --max_classes 0 \
  --cache_dir tfmplayground/priors/real_data/data/cache \
  --train_pool tfmplayground/priors/real_data/data/pools/train_pool_regression.txt \
  --mode only
```

### Mixed target-column mode with fallback pool

Use this when you want to sample target columns from broad datasets (`train_pool_all.txt`) while keeping episode task type fixed.

```bash
python -m tfmplayground.priors \
  --lib real \
  --prior_type mlp \
  --max_classes 0 \
  --cache_dir tfmplayground/priors/real_data/data/cache \
  --train_pool tfmplayground/priors/real_data/data/pools/train_pool_all.txt \
  --mode mixed \
  --fallback_pool tfmplayground/priors/real_data/data/pools/train_pool_regression.txt
```

Notes:
- Task type is inferred from `--max_classes`:
  - `> 0` -> classification
  - `0` -> regression
- In `mode=mixed`, the generator enforces target type strictly.
- If no valid target column is found in primary/fallback pools, it raises an error (it does not silently switch task type).

## Data Layout

Expected cache and pool paths:

```text
tfmplayground/priors/real_data/data/
  cache/
    datasets/openml_<dataset_id>.npz
    metadata.json
  pools/
    train_pool_all.txt
    train_pool_classification.txt
    train_pool_regression.txt
    report_<...>.json
```
