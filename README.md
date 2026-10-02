<div align="center">

<img src="https://salihboraozturk.com/other/images/tfmp.png" width="200" alt="TFM-Playground">

# TFM-Playground

**Many tabular foundation models. One interface. Training and inference.**

[![python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/release/python-3120/)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

</div>

A fully open source playground for tabular foundation models: five architectures behind one interface, with priors, a training loop and an evaluation pipeline. It is a starting point for students who want to see how these models work, and a base for research on top of them.

### Install

```
pip install uv
git clone https://github.com/automl/TFM-Playground.git
cd TFM-Playground
uv sync
```

### Quickstart

This repository has no checkpoints yet.

```python
model = pretrainTFM(
    problem="classification",
    model=NanoTabPFNModel(config=NanoTabPFNClassifierConfig()),
    prior=NanoTabICLPrior(
        config=NanoTabICLClassificationPriorConfig(
            min_num_datapoints=50,
            max_num_datapoints=50,
            max_num_features=3,
            max_num_classes=3,
        )
    ),
    training=ClassificationTrainingConfig(batch_size=8, steps=25, epochs=12),
)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

predictions = TabularClassifier(model).fit(X_train, y_train).predict(X_test)
```

That run takes about 5 minutes on a laptop cpu, on tables of 50 rows, 3 features and up to 3 classes. The mean roc auc over the toy tasks goes from 0.67 after the first epoch to 0.98 after the twelfth.

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

A prior also writes to a dataset dump that you can train from later:

```
uv run python -m tfmplayground.priors --lib tabicl \
       --prior_type mix_scm \
       --num_batches 1000 --batch_size 4 \
       --min_features 3 --max_features 3 \
       --max_seq_len 50 --max_classes 3 \
       --save_path tabicl_4k_50x3.h5
```

### Citation

```bibtex
@misc{tfmplayground2026,
  title        = {TFM-Playground},
  author       = {The TFM-Playground Authors},
  year         = {2026},
  howpublished = {\url{https://github.com/automl/TFM-Playground}}
}
```

### License

Apache 2.0, see [LICENSE](LICENSE). The vendored files name their own source and license at the top.
