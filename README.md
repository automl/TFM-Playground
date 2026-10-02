<div align="center">

<img src="https://salihboraozturk.com/other/images/tfmp.png" width="200" alt="TFM-Playground">

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

That last command runs [examples/pretraining_quickstart.py](examples/pretraining_quickstart.py). It trains a toy classifier on toy tables in about 5 minutes on a CPU, from swappable configurations like these:

```python
modelconfig = ...
priorconfig = ...
evalconfig = ...
trainconfig = ...
experimentconfig= ...

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

<details>
<summary>Each one is an adapter over upstream code, with a classifier and a regressor config.</summary>

- [adapter](tfmplayground/models/nanotabpfn.py) · [config](tfmplayground/configs/models.py) - nanotabpfn - [repo](https://github.com/automl/nanoTabPFN) · [paper](https://arxiv.org/abs/2511.03634)
- [adapter](tfmplayground/models/moddednanotabpfn.py) · [config](tfmplayground/configs/models.py) - moddednanotabpfn - [repo](https://github.com/borawhocodess/modded-nanotabpfn) · [paper](https://arxiv.org/abs/2606.03681)
- [adapter](tfmplayground/models/nanotabicl.py) · [config](tfmplayground/configs/models.py) - nanotabicl - [repo](https://github.com/soda-inria/nanotabicl)
- [adapter](tfmplayground/models/tabicl.py) · [config](tfmplayground/configs/models.py) - tabicl - [repo](https://github.com/soda-inria/tabicl) · [paper](https://arxiv.org/abs/2602.11139)
- [adapter](tfmplayground/models/tabfm.py) · [config](tfmplayground/configs/models.py) - tabfm - [repo](https://github.com/google-research/tabfm) · [paper](https://arxiv.org/abs/2609.37959)

</details>

### Priors

Priors generate the synthetic tables for pretraining.

<details>
<summary>On the fly, where each batch is sampled when the training loop asks for it</summary>

- [adapter](tfmplayground/priors/nanotabicl.py) · [config](tfmplayground/configs/priors.py) - nanotabicl - [repo](https://github.com/soda-inria/nanotabicl)
- [adapter](tfmplayground/priors/tabicl.py) · [config](tfmplayground/configs/priors.py) - tabicl - [repo](https://github.com/soda-inria/tabicl) · [paper](https://arxiv.org/abs/2602.11139)

</details>

<details>
<summary>Dumping, where the tables are written once and read back later</summary>

Priors can also be written to a dump that you can use later:

```
uv run python -m tfmplayground.priors --lib tabicl --prior_type mix_scm --num_batches 1000 --batch_size 4 --max_classes 3 --max_seq_len 50 --min_features 3 --max_features 3 --save_path dump-d1000b4r50c3-3-tabicl.h5
```

These priors can be dumped:

- [adapter](tfmplayground/priors/ticl.py) - ticl - [repo](https://github.com/microsoft/ticl)
- [adapter](tfmplayground/priors/tabicl.py) - tabicl - [repo](https://github.com/soda-inria/tabicl) · [paper](https://arxiv.org/abs/2602.11139)
- [adapter](tfmplayground/priors/tabpfn.py) - tabpfn - [repo](https://github.com/automl/tabpfn-v1-prior)

And use later with:

- [adapter](tfmplayground/priors/dump.py) · [config](tfmplayground/configs/priors.py) - dump

</details>

### Citation

```bibtex
@misc{tfmplayground2026,
  title        = {TFM-Playground},
  author       = {TFM-Playground Authors},
  year         = {2026},
  howpublished = {\url{https://github.com/automl/TFM-Playground}}
}
```
