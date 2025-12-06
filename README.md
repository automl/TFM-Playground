# TFM-Playground

The purpose of this repository is to provide a fully open source playground for tabular foundation models.
It contains a much smaller and simpler implementation of the TabPFNv2 architecture (nanoTabPFN) as well as a training loop, multiple interfaces to load prior data and an evaluation pipeline. We are planning to rapidly extend the repository with more features, prior interfaces and architectures.
It is supposed to be a good starting point for students and researchers that are interested in learning about how Tabular foundation models work under the hood.

Clone the repository, afterwards install dependencies via:
```
pip install -e .
```

We offer the same interface as TabPFN:
```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tfmplayground import NanoTabPFNClassifier

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize a classifier
clf = NanoTabPFNClassifier()
clf.fit(X_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

# Predict labels
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))
```

### Our Code

`tfmplayground/model.py` contains the implementation of the architecture in less than 250 lines of code. `tfmplayground/train.py` implements a simple training loop in under 100 lines and `tfmplayground/priors.py` implements a dataloader that allows you to load a dump pre-generated from a prior.
We will release multiple dumps of different scales soon. We also offer an interface where you can provide your own get\_batch function.

### Pretrain your own small nanoTabPFN
First we download 100k pre-generated datasets with 50 datapoints, 3 features and up to 3 classes each from [here](https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/50x3_3_100k_classification.h5).

Then you can run:
```
python pretrain_classification.py --epochs 80 --steps 25 --batchsize 50 --priordump 50x3_3_100k_classification.h5
```
This should take less than 5 min on a modern NVIDIA GPU (around 10 minutes on Macbook M4 Pro GPU and around 40 min on M4 Pro CPU).

We also offer a pre-generated dataset containing 1.28M tables with 50 datapoints and 3 features each for regression [here](https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/50x3_1280k_regression.h5).

You can pretrain on it using `python pretrain_regressor.py`.

#### Step by Step Explanation (Classifier)

First we import our Architecture, Prior interface and training loop, etc.
```python
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors import PriorDumpDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device
from tfmplayground.interface import NanoTabPFNClassifier
from tfmplayground.callbacks import ConsoleLoggerCallback

from torch.nn import CrossEntropyLoss
```
then we instantiate our model and loss criterion:
```python
model = NanoTabPFNModel(
    num_attention_heads=6,
    embedding_size=192,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=10,
)
criterion = CrossEntropyLoss()
```
then we instantiate our prior:
```python
device = get_default_device()
prior = PriorDumpDataLoader(filename='50x3_3_100k_classification.h5', num_steps=25, batch_size=50, device=device)
```
and finally train our model:
```python
trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=criterion,
    epochs=80,
    device=device,
    callbacks=[ConsoleLoggerCallback()]
)
```

### Creating your own datasets
Check out [tfmplayground.priors](https://github.com/automl/TFM-Playground/tree/main/tfmplayground/priors) to create your own data using publicly available priors.

You can use tfmplayground.priors as a command-line-tool to pre-generate data from a prior, e.g. via
```
python -m tfmplayground.priors --lib tabicl \
       --num_batches 1000 --batch_size 4 \
       --min_features 3 --max_features 3 \
       --max_seq_len 50 --max_classes 3 \
       --save_path tabicl_4k_50x3.h5
```
which can afterwards be loaded via
```python
from tfmplayground.priors.dataloader import PriorDumpDataLoader
prior = PriorDumpDataLoader('tabicl_4k_50x3.h5', num_steps=20, batch_size=4, device='cpu')
```
You can also just let it create the data on-the-fly via:
```python
from tfmplayground.priors.dataloader import TabICLPriorDataLoader
prior = TabICLPriorDataLoader(
    num_steps=20,
    batch_size=4,
    num_datapoints_max=50,
    min_features=3,
    max_features=3,
    max_num_classes=3,
    device='cpu'
)
```
You can check out `next(iter(prior))` if you want to see an example batch.

Check out `visualization_demo.ipynb` for some more examples.

### Supported Priors

- [TabICL](https://github.com/soda-inria/tabicl) (Classification)
- [TICL](https://github.com/microsoft/ticl) (Regression, Classification)
