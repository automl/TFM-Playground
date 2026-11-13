from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from tfmplayground import TabularRegressor, pretrainTFM
from tfmplayground.configs.models import NanoTabPFNRegressorConfig
from tfmplayground.configs.priors import NanoTabICLRegressionPriorConfig
from tfmplayground.configs.training import RegressionTrainingConfig
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors import NanoTabICLPrior

model = pretrainTFM(
    problem="regression",
    model=NanoTabPFNModel(config=NanoTabPFNRegressorConfig(head="scalar", num_outputs=1)),
    prior=NanoTabICLPrior(
        config=NanoTabICLRegressionPriorConfig(
            min_num_datapoints=50,
            max_num_datapoints=50,
            max_num_features=3,
        )
    ),
    training=RegressionTrainingConfig(batch_size=2, steps=100, epochs=10),
)

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

predictions = TabularRegressor(model).fit(X_train, y_train).predict(X_test)
