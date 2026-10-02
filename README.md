<div align="center">

<img src="https://salihboraozturk.com/other/images/tfmp.png" width="200" alt="TFM-Playground">

# TFM-Playground

[![python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/release/python-3120/)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

</div>

The purpose of this repository is to provide a fully open source playground for tabular foundation models. It holds five model architectures behind one interface, together with live and dumped priors, a training loop, an evaluation pipeline and experiment tracking. It is supposed to be a good starting point for students and researchers that are interested in learning about how tabular foundation models work under the hood.

### Install

```
git clone https://github.com/automl/TFM-Playground.git
cd TFM-Playground
uv sync
```

### Quickstart

This repository has no checkpoint, so you pretrain a small model first, and then predict with it.

```python
model = pretrainTFM(
    problem="classification",
    model=NanoTabPFNModel(config=NanoTabPFNClassifierConfig()),
    prior=NanoTabICLPrior(config=NanoTabICLClassificationPriorConfig()),
)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

predictions = TabularClassifier(model).fit(X_train, y_train).predict(X_test)
```

One [example](examples) per problem with more configurations:

```
uv run python examples/pretraining_classification.py
uv run python examples/pretraining_regression.py
```

### Models

- nanotabpfn - [adapter](tfmplayground/models/nanotabpfn.py) · [config](tfmplayground/configs/models.py) · [repo](https://github.com/automl/nanoTabPFN) · [paper](https://arxiv.org/abs/2511.03634)
- moddednanotabpfn - [adapter](tfmplayground/models/moddednanotabpfn.py) · [config](tfmplayground/configs/models.py) · [repo](https://github.com/borawhocodess/modded-nanotabpfn) · [paper](https://arxiv.org/abs/2606.03681)
- nanotabicl - [adapter](tfmplayground/models/nanotabicl.py) · [config](tfmplayground/configs/models.py) · [repo](https://github.com/soda-inria/nanotabicl)
- tabicl - [adapter](tfmplayground/models/tabicl.py) · [config](tfmplayground/configs/models.py) · [repo](https://github.com/soda-inria/tabicl) · [paper](https://arxiv.org/abs/2602.11139)
- tabfm - [adapter](tfmplayground/models/tabfm.py) · [config](tfmplayground/configs/models.py) · [repo](https://github.com/google-research/tabfm) · [paper](https://arxiv.org/abs/2609.37959)

### Priors

- nanotabicl - [adapter](tfmplayground/priors/nanotabicl.py) · [config](tfmplayground/configs/priors.py) · [repo](https://github.com/soda-inria/nanotabicl)
- tabicl - [adapter](tfmplayground/priors/tabicl.py) · [config](tfmplayground/configs/priors.py) · [repo](https://github.com/soda-inria/tabicl) · [paper](https://arxiv.org/abs/2602.11139)
- dump - [adapter](tfmplayground/priors/dump.py) · [config](tfmplayground/configs/priors.py)

Also for dumping:

```
uv run python -m tfmplayground.priors --lib tabicl \
       --prior_type mix_scm \
       --num_batches 1000 --batch_size 4 \
       --min_features 3 --max_features 3 \
       --max_seq_len 50 --max_classes 3 \
       --save_path tabicl_4k_50x3.h5
```

### License

Apache 2.0, see [LICENSE](LICENSE). The vendored files name their own source and license at the top.
