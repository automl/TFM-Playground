from tfmplayground.configs.evaluation import EvaluationConfig
from tfmplayground.configs.models import TabFMRegressorConfig
from tfmplayground.configs.priors import TabICLRegressionPriorConfig
from tfmplayground.configs.training import (
    RegressionExperimentConfig,
    RegressionTrainingConfig,
)
from tfmplayground.models.tabfm import TabFMModel
from tfmplayground.priors import TabICLPrior
from tfmplayground.training.pretrain import pretrainTFM
from tfmplayground.utils import get_default_device, set_randomness_seed

device = get_default_device()
set_randomness_seed(11)  # important for making model weight initialization reproducible

model_config = TabFMRegressorConfig()
prior_config = TabICLRegressionPriorConfig(num_datapoints_min=10, num_datapoints_max=20, num_features_max=5)
eval_config = EvaluationConfig(tasks="tabarena", max_n_samples=1200)
train_config = RegressionTrainingConfig(seed=11, batch_size=2, epochs=3, steps=1)
experiment_config = RegressionExperimentConfig(name="regression_example")

model = pretrainTFM(
    problem="regression",
    model=TabFMModel(config=model_config),
    prior=TabICLPrior(config=prior_config, device=device),
    eval=eval_config,
    training=train_config,
    experiment=experiment_config,
)
