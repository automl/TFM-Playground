<div align="center">

<img src="https://ml.informatik.uni-freiburg.de/research-artifacts/TFM-Playground/tfmplayground-logo.png" width="200" alt="TFM-Playground">

# TFM-Playground

**Tabular Foundation Models and Priors. One Interface. Training and Inference.**

[![python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/release/python-3120/)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

</div>

A fully open source playground for tabular foundation models: many architectures behind one interface, with many priors, a pretraining loop and an evaluation pipeline. It is a starting point for anyone who wants to see how these models work, and a base for research on top of them.

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
modelconfig = ...
priorconfig = ...
evalconfig = ...
trainconfig = ...
experimentconfig = ...

model = pretrainTFM(
    problem="classification",
    model=...Model(config=modelconfig),
    prior=...Prior(config=priorconfig),
    eval=evalconfig,
    training=trainconfig,
    experiment=experimentconfig,
)
```

Fully configurable examples are in [examples](examples):

```
uv run python examples/pretraining_classification.py
uv run python examples/pretraining_regression.py
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

```bibtex
@misc{tfmplayground2026,
  title        = {TFM-Playground},
  author       = {TFM-Playground Authors},
  year         = {2026},
  howpublished = {\url{https://github.com/automl/TFM-Playground}}
}
```
