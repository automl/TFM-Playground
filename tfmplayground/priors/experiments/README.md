# Prior Experiments

This folder contains the experiment scripts to:
- generate prior datasets
- train one model per prior
- compare saved models afterwards

## Quick start

In the project root:

```bash
pip install -e .
```

Edit the experiment config in [config.yaml](config.yaml).

You can use the interactive launcher:

```bash
sh tfmplayground/priors/experiments/run.sh classification
# or
sh tfmplayground/priors/experiments/run.sh regression
```

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

## Notes

- Omitting `--priors` keeps the interactive selection behavior.
- If you add a new prior, expose it in [config.yaml](config.yaml) so the scripts can discover it.
