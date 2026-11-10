<div align="center">

<img src="https://ml.informatik.uni-freiburg.de/research-artifacts/TFM-Playground/tfmplayground-logo.png" width="200" alt="TFM-Playground">

# TFM-Playground

**Tabular Foundation Models and Priors. One Interface. Training and Inference.**

[![python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/release/python-3120/)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

</div>

A fully open source playground for tabular foundation models: many architectures behind one interface, with many priors, a pretraining loop and an evaluation pipeline. It is a starting point for anyone who wants to develop or research tabular foundation models. If you are a new to tabular foundation models we recommend checking out [nanoTabPFN](https://github.com/automl/nanoTabPFN) first as an introduction.

### Quickstart

```
pip install uv
git clone https://github.com/automl/TFM-Playground.git
cd TFM-Playground
uv sync
uv run python examples/pretraining_quickstart.py
```

That last command runs [examples/pretraining_quickstart.py](examples/pretraining_quickstart.py). It trains a toy classifier on toy tables in about 5 minutes on a laptop, from swappable configurations like these:

```python
model_config = ...
prior_config = ...
eval_config = ...
train_config = ...
experiment_config = ...

model = pretrainTFM(
    problem="classification",
    model=...Model(config=model_config),
    prior=...Prior(config=prior_config),
    eval=eval_config,
    training=train_config,
    experiment=experiment_config,
)
```

Fully configurable examples are in [examples](examples):

```
uv run python examples/pretraining_classification.py
uv run python examples/pretraining_regression.py
```

The same runs can be driven from yaml instead of Python, see [Configuration](#configuration):

```
uv run python -m tfmplayground.training --config-name debug      # tiny model, tiny tables, a few steps
```

### Configuration

There are two ways to configure a run, and they describe the same thing:

- **In Python**, build the config dataclasses from [tfmplayground/configs](tfmplayground/configs) and pass models and
  priors to `pretrainTFM`, as the [examples](examples) do. This is the way to use the package as a library.
- **From yaml**, compose a run from the files in [yaml_configs](yaml_configs) and start it with
  `python -m tfmplayground.training`. This is the way to run experiments in a clone of the repo.

The yaml files never define configuration on their own: each one refines one of the dataclasses, so every field is
checked against it and unknown fields are rejected. The tree has run configs at the top level and one directory per
part of a run:

```
yaml_configs/
  classification.yaml  regression.yaml  quickstart.yaml  debug.yaml     # run configs
  model/       nanotabpfn_classifier.yaml  tabfm_regressor.yaml  ...     # one file per model and problem
  prior/       nanotabicl_classification.yaml  tabicl_regression.yaml ...
  training/    classification.yaml  regression.yaml  *_debug.yaml
  experiment/  classification.yaml  regression.yaml
  eval/        toy.yaml  tabarena.yaml
```

A run config picks one file from each directory; the command line can swap any of them or set any field,
using [Hydra's override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/):

```
uv run python -m tfmplayground.training --config-name classification          # the run config to start from
uv run python -m tfmplayground.training --config-name regression model=tabfm_regressor prior=tabicl_regression
uv run python -m tfmplayground.training --config-name classification training.epochs=2 model.num_layers=4
```

`group=option` swaps the file `yaml_configs/<group>/<option>.yaml` in, `group.field=value` sets one field. Files
named plainly, like `tabicl_classifier`, keep the dataclass defaults, which are the upstream model sizes; `*_small`,
`*_tiny` and `*_debug` variants carry laptop sizes.

To keep your own setup, copy a file and change fields in it. A model variant:

```yaml
# yaml_configs/model/tabicl_classifier_wide.yaml
defaults:
  - TabICLClassifierConfig   # the dataclass this file refines
embed_dim: 256
```

A run config that reuses another and swaps parts, as `debug.yaml` does:

```yaml
defaults:
  - classification
  - override model: tabicl_classifier_wide
  - _self_
experiment:
  name: wide
```

Every run also writes its full configuration as `<run id>-config.yaml` into its experiment directory, next to the
log and the checkpoints. That file composes as is, so a finished run can be repeated or tweaked:

```
uv run python -m tfmplayground.training --config-dir workdir/experiments/classification/debug/<run id> --config-name <run id>-config training.epochs=5
```

### Models

Each one is an adapter over upstream code, with a classifier and a regressor config.

- nanotabpfn - [adapter](tfmplayground/models/nanotabpfn.py) · [config](tfmplayground/configs/models.py) · [repo](https://github.com/automl/nanoTabPFN) · [paper](https://arxiv.org/abs/2511.03634)
- moddednanotabpfn - [adapter](tfmplayground/models/moddednanotabpfn.py) · [config](tfmplayground/configs/models.py) · [repo](https://github.com/borawhocodess/modded-nanotabpfn) · [paper](https://arxiv.org/abs/2606.03681)
- nanotabicl - [adapter](tfmplayground/models/nanotabicl.py) · [config](tfmplayground/configs/models.py) · [repo](https://github.com/soda-inria/nanotabicl)
- tabicl - [adapter](tfmplayground/models/tabicl.py) · [config](tfmplayground/configs/models.py) · [repo](https://github.com/soda-inria/tabicl) · [paper](https://arxiv.org/abs/2602.11139)
- tabfm - [adapter](tfmplayground/models/tabfm.py) · [config](tfmplayground/configs/models.py) · [repo](https://github.com/google-research/tabfm) · [paper](https://arxiv.org/abs/2609.37959)

### Priors

Priors generate the synthetic tables for pretraining.

#### On the fly

Each batch is sampled when the training loop asks for it.

- nanotabicl - [adapter](tfmplayground/priors/nanotabicl.py) · [config](tfmplayground/configs/priors.py) · [repo](https://github.com/soda-inria/nanotabicl)
- tabicl - [adapter](tfmplayground/priors/tabicl.py) · [config](tfmplayground/configs/priors.py) · [repo](https://github.com/soda-inria/tabicl) · [paper](https://arxiv.org/abs/2602.11139)

#### Dumping

Priors can also be written to a dump that you can use later:

```
uv run python -m tfmplayground.priors --lib tabicl --prior_type mix_scm --num_batches 1000 --batch_size 4 --max_classes 3 --max_seq_len 50 --min_features 3 --max_features 3 --save_path dump-d1000b4r50c3-3-tabicl.h5
```

These priors can be dumped:

- ticl - [adapter](tfmplayground/priors/ticl.py) · [repo](https://github.com/microsoft/ticl)
- tabicl - [adapter](tfmplayground/priors/tabicl.py) · [repo](https://github.com/soda-inria/tabicl) · [paper](https://arxiv.org/abs/2502.05564)
- tabpfn - [adapter](tfmplayground/priors/tabpfn.py) · [repo](https://github.com/automl/tabpfn-v1-prior)

Load a dump with:

- dump - [adapter](tfmplayground/priors/dump.py) · [config](tfmplayground/configs/priors.py)

### Citation

If you use the TFM-Playground please cite:

```bibtex
@misc{pfefferle2026tfmplayground,
  title={TFM-Playground: A Playground for Tabular Foundation Models},
  author={Pfefferle, Alexander and Hog, Johannes and Öztürk, Salih Bora and Kaya, Kürşat and Türkmen, Zeynep and Hutter, Frank},
  year={2026},
  url={https://github.com/automl/TFM-Playground/}
}
```

and if you are using the nanoTabPFN-Model specifically please also cite:

```bibtex
@article{pfefferle2025nanotabpfn,
  title={nanoTabPFN: A Lightweight and Educational Reimplementation of TabPFN},
  author={Pfefferle, Alexander and Hog, Johannes and Purucker, Lennart and Hutter, Frank},
  journal={arXiv preprint arXiv:2511.03634},
  year={2025}
}
```
