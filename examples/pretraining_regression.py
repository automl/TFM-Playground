from tfmplayground.configs.evaluation import EvaluationConfig
from tfmplayground.configs.models import NanoTabPFNRegressorConfig
from tfmplayground.configs.priors import NanoTabICLRegressionPriorConfig
from tfmplayground.configs.training import (
    RegressionExperimentConfig,
    RegressionTrainingConfig,
)
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors import NanoTabICLPrior
from tfmplayground.training.pretrain import pretrainTFM
from tfmplayground.utils import get_default_device, set_randomness_seed

device = get_default_device()
set_randomness_seed(11)  # important for making model weight initialization reproducible

model_config = NanoTabPFNRegressorConfig(num_layers=10, num_mem_chunks=8, head="scalar", num_outputs=1)
prior_config = NanoTabICLRegressionPriorConfig(max_num_datapoints=2048, min_num_features=5, max_num_features=32)
eval_config = EvaluationConfig(tasks="tabarena", max_n_samples=50000, max_n_features=200)
train_config = RegressionTrainingConfig(seed=11, batch_size=2, epochs=100, steps=1000)
experiment_config = RegressionExperimentConfig(name="regression_example")

model = pretrainTFM(
    problem="regression",
    model=NanoTabPFNModel(config=model_config),
    prior=NanoTabICLPrior(config=prior_config, device=device),
    eval=eval_config,
    training=train_config,
    experiment=experiment_config,
)
