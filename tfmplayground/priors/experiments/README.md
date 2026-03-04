# Prior Experiments

This folder contains scripts to generate priors, train per-prior models, and run analysis/comparison pipelines for classification and regression.

## Quick start
0) Arrange the environment
```bash
# In the project root, using a virtual env:
pip install -e .
```

1) Edit the experiment [config](./config.yaml).

2) Run the experiment launcher.

```bash
sh run.sh classification
# or
sh run.sh regression
```

You will be prompted to choose from available priors and whether you want analysis and comparison analysis.

### Non-interactive / SLURM mode

All scripts accept `--priors` to bypass interactive prompts.

**Data generation** — pick exactly which priors to generate:

```bash
# All priors
python generate_data.py --mode regression --priors all

# Specific priors
python generate_data.py --mode regression --priors ticl_gp ticl_mlp tabpfn_mlp
```

**Training** — pick exactly which priors to train on:

```bash
# All available priors
python train_models.py --problem_type regression --priors all --epochs 50 --steps 100

# Specific priors (great for SLURM job arrays)
python train_models.py --problem_type regression --priors ticl_gp --epochs 50
```

**Via run.sh** — pass `--priors` after the mode:

```bash
# Non-interactive data generation (analysis step stays interactive)
sh run.sh regression --priors ticl_gp ticl_mlp tabpfn_mlp
```

> **Note:** Omitting `--priors` from any command keeps the original interactive behavior.

## Compare trained models

Use the comparison script to run a side-by-side analysis using saved priors.

```bash
python compare_models.py \
  --prior ./classification/results/data/prior_tabicl_mlp_scm_10x8_50x3.h5 \
  --prior ./classification/results/data/prior_ticl_classification_adapter_10x8_50x3.h5 \
  --prior ./classification/results/data/prior_tabpfn_mlp_10x8_50x3.h5
```

## Outputs and where to look

- Generated prior data: [classification/results/data](classification/results/data) and [regression/results/data](regression/results/data)
- Reports: [classification/results/reports](classification/results/reports) and [regression/results/reports](regression/results/reports)
- Plots: [plots](plots) and [plots](plots)
- Checkpoints and working files: [workdir](workdir)

## Configuration notes

- The primary configuration lives in [config.yaml](config.yaml).
- If you add new priors, make sure they are exposed in the config so the launcher can prompt for them.

