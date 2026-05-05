# Prior Experiments

This folder contains the experiment scripts to:
- generate prior datasets
- analyze the generated data
- train one model per prior
- compare saved models afterwards

## Files

- `generate_data.py`: creates `.h5` prior dumps via the main `tfmplayground.priors` CLI
- `analyze_priors.py`: interactive analysis of generated `.h5` files (distributions, feature importance, etc.)
- `train_models.py`: trains one nanoTabPFN model per prior, saves checkpoints + metadata
- `compare_models.py`: loads trained models and generates comparison plots, metrics, and TabArena evaluation
- `compare_time_tradeoff.py`: analyzes prior generation cost vs downstream performance
- `experiment_evaluation.py`: shared evaluation helpers (OpenML predictions, task ID lists)
- `base_analyzer.py`: base class for data analyzers
- `config.yaml`: all experiment settings (priors, hyperparameters, output paths)

## Quick Start

In the project root:

```bash
pip install -e .
```

Edit the experiment config in [config.yaml](config.yaml).

## Generate Data

Use `generate_data.py` to create `.h5` prior dumps for one problem type.

```bash
# All configured priors
python tfmplayground/priors/experiments/generate_data.py \
  --mode regression \
  --priors all

# Selected priors
python tfmplayground/priors/experiments/generate_data.py \
  --mode classification \
  --priors ticl_gp tabpfn_mlp
```

Generated files are written to:
- [classification/results/data](classification/results/data)
- [regression/results/data](regression/results/data)

Pay attention to:
- `--mode` must match the experiment you want to train later.
- Generated filenames depend on the config. If you change data-generation settings, check the existing `.h5` files before reusing them.
- Re-generating with the same output path overwrites the existing prior dump.

## Analyze Data

Use `analyze_priors.py` to inspect the generated `.h5` files. This is always interactive.

```bash
python tfmplayground/priors/experiments/analyze_priors.py \
  --mode regression
```

The tool lets you select which priors to analyze and whether to run individual reports, pairwise comparisons, or both.

## Train Models

Use `train_models.py` after the `.h5` files exist.

```bash
# All discovered priors
python tfmplayground/priors/experiments/train_models.py \
  --problem_type regression \
  --priors all \
  --epochs 50 \
  --steps 100

# One prior
python tfmplayground/priors/experiments/train_models.py \
  --problem_type classification \
  --priors ticl_gp \
  --epochs 100
```

Saved models are written to:
- [classification/results/trained_models](classification/results/trained_models)
- [regression/results/trained_models](regression/results/trained_models)

Each prior gets a stable folder containing:
- `model.pth`
- `metadata.json`
- `latest_checkpoint.pth`
- `bucket_edges.pth` for regression

Pay attention to:
- `--problem_type` must match the problem type of the prior dump.
- `--epochs` is the target total epoch count. Re-running with a larger value resumes from `latest_checkpoint.pth`.
- Training now reuses the same prior folder instead of creating timestamped folders.
- If you change config in a way that should start a fresh run, check or remove the existing trained-model folder for that prior first. Otherwise the script may resume from the old checkpoint and append to old metadata.

## Compare Models

```bash
python tfmplayground/priors/experiments/compare_models.py \
  --problem_type classification
```

Or use a custom trained-model directory:

```bash
python tfmplayground/priors/experiments/compare_models.py \
  --problem_type regression \
  --models_dir /path/to/trained_models
```

Additional flags:
- `--models all` or `--models tabpfn_mlp ticl_gp` for non-interactive model selection.
- `--skip_tabarena` to skip the TabArena datasets evaluation.
- `--skip_data_similarity` to skip prior data-similarity analysis.
- `--tabarena_cache_dir` to set a custom OpenML data cache directory.

## Compare Time Trade-offs

Use `compare_time_tradeoff.py` to analyze prior generation cost against downstream performance.

```bash
python tfmplayground/priors/experiments/compare_time_tradeoff.py \
  --problem_type classification

# With specific priors
python tfmplayground/priors/experiments/compare_time_tradeoff.py \
  --problem_type regression \
  --priors tabpfn_mlp ticl_gp
```

Additional flags:
- `--baseline_prior` to set a reference prior for improvement calculations.
- `--exclude_real_priors` to filter out real data priors from plots.
- `--allow_missing_generation_time` to handle priors without timing metadata.

## Notes

- Omitting `--priors` (or `--models` for compare) keeps the interactive selection behavior.
- If you add a new prior, expose it in [config.yaml](config.yaml) so the scripts can discover it.

