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
    prior=NanoTabICLPrior(
        config=NanoTabICLClassificationPriorConfig(
            min_num_datapoints=50,
            max_num_datapoints=50,
            max_num_features=3,
            max_num_classes=3,
        )
    ),
    training=ClassificationTrainingConfig(batch_size=2, steps=100, epochs=10),
)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

predictions = TabularClassifier(model).fit(X_train, y_train).predict(X_test)
