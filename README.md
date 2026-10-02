# TFM-Playground

[![python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/release/python-3120/)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

The purpose of this repository is to provide a fully open source playground for tabular foundation models. It holds five model architectures behind one interface, together with live and dumped priors, a training loop, an evaluation pipeline and experiment tracking. You pretrain every model with the same function, on any prior, for classification or regression, and you predict with every model through the same two estimator classes. It is supposed to be a good starting point for students and researchers that are interested in learning about how tabular foundation models work under the hood.

### install

The project needs python 3.12, and it gets three of its dependencies from git.

```
git clone https://github.com/automl/TFM-Playground.git
cd TFM-Playground
uv sync
```

`uv sync` makes the environment, takes a python 3.12, and installs the project in editable mode. `uv run` then runs a command inside it, with no activation.

### quickstart

This repository has no checkpoint, so you pretrain a small model first, and then predict with it.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tfmplayground import TabularClassifier, pretrainTFM
from tfmplayground.configs.models import NanoTabPFNClassifierConfig
from tfmplayground.configs.priors import NanoTabICLClassificationPriorConfig
from tfmplayground.configs.training import ClassificationTrainingConfig
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors import NanoTabICLPrior

model = pretrainTFM(
    problem="classification",
    model=NanoTabPFNModel(config=NanoTabPFNClassifierConfig()),
    prior=NanoTabICLPrior(config=NanoTabICLClassificationPriorConfig(max_num_features=5)),
    training=ClassificationTrainingConfig(batch_size=1, steps=2, epochs=2),
)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

classifier = TabularClassifier(model)
classifier.fit(X_train, y_train)
probabilities = classifier.predict_proba(X_test)
```

The run prints one line per epoch:

```
experiment: 261002-141506-26c5a21c-test
epoch 1 | epoch time 0.86s | mean loss 2.11 | mean roc auc 0.63 | tasks 3
epoch 2 | epoch time 0.73s | mean loss 1.97 | mean roc auc 0.67 | tasks 3
```

Two epochs take about 10 seconds on a cpu, and the model that they give is still near random. The two files in [examples](examples) show settings that make sense:

```
uv run python examples/pretraining_classification.py
uv run python examples/pretraining_regression.py
```

### pretraining

`pretrainTFM` takes one model, one prior and four configs. Every config is a dataclass, and every argument but the first two has a default.

| argument | config | holds |
| --- | --- | --- |
| `problem` | - | `classification` or `regression` |
| `model` | `configs/models.py` | architecture sizes and the number of outputs |
| `prior` | `configs/priors.py` | table sizes, feature counts and class counts |
| `training` | `configs/training.py` | seed, learning rate, batch size, steps, epochs, gradient clip |
| `eval` | `configs/evaluation.py` | tasks and their size limits |
| `experiment` | `configs/training.py` | run name and the directory for the artifacts |
| `device` | - | the device for the model and the batches |

The function picks the loss from the problem: cross entropy for classification, and for regression the head that the model config names. A `buckets` head fits its bucket borders from the prior before the run starts.

### models

Every model takes `X_train`, `y_train` and `X_test`, and predicts the test rows with the train rows as context.

- nanotabpfn - [adapter](tfmplayground/models/nanotabpfn.py) · [config](tfmplayground/configs/models.py) · [repo](https://github.com/automl/nanoTabPFN) · [paper](https://arxiv.org/abs/2511.03634)
- moddednanotabpfn - [adapter](tfmplayground/models/moddednanotabpfn.py) · [config](tfmplayground/configs/models.py) · [repo](https://github.com/borawhocodess/modded-nanotabpfn) · [paper](https://arxiv.org/abs/2606.03681)
- nanotabicl - [adapter](tfmplayground/models/nanotabicl.py) · [config](tfmplayground/configs/models.py) · [repo](https://github.com/soda-inria/nanotabicl)
- tabicl - [adapter](tfmplayground/models/tabicl.py) · [config](tfmplayground/configs/models.py) · [repo](https://github.com/soda-inria/tabicl) · [paper](https://arxiv.org/abs/2602.11139)
- tabfm - [adapter](tfmplayground/models/tabfm.py) · [config](tfmplayground/configs/models.py) · [repo](https://github.com/google-research/tabfm) · [blog](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)

### priors

A prior gives one batch of tables, already split into a train part and a test part. The nanotabicl and the tabicl priors sample live tables, and the dump prior reads an h5 file and starts again at its end.

- nanotabicl - [adapter](tfmplayground/priors/nanotabicl.py) · [config](tfmplayground/configs/priors.py) · [repo](https://github.com/soda-inria/nanotabicl)
- tabicl - [adapter](tfmplayground/priors/tabicl.py) · [config](tfmplayground/configs/priors.py) · [repo](https://github.com/soda-inria/tabicl) · [paper](https://arxiv.org/abs/2602.11139)
- dump - [adapter](tfmplayground/priors/dump.py) · [config](tfmplayground/configs/priors.py)

You can look at one batch:

```python
from tfmplayground.configs.priors import NanoTabICLClassificationPriorConfig
from tfmplayground.priors import NanoTabICLPrior

prior = NanoTabICLPrior(config=NanoTabICLClassificationPriorConfig())
x_train, y_train, x_test, y_test = prior.batch(batch_size=2)
```

Two dumps are available for download. The [classification dump](https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/50x3_3_100k_classification.h5) holds 100k tables of 50 rows, 3 features and up to 3 classes each, at 0.1 GB. The [regression dump](https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/50x3_1280k_regression.h5) holds 1.28M tables of 50 rows and 3 features each, at 1.0 GB.

The priors package also writes dumps of its own, from the ticl, tabicl and tabpfn libraries:

```
uv run python -m tfmplayground.priors --lib tabicl \
       --prior_type mix_scm \
       --num_batches 1000 --batch_size 4 \
       --min_features 3 --max_features 3 \
       --max_seq_len 50 --max_classes 3 \
       --save_path tabicl_4k_50x3.h5
```

### license

Apache 2.0, see [LICENSE](LICENSE). The vendored files name their own source and license at the top.
