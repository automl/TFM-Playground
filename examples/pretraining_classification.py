# %% imports
from tfmplayground.configs.evaluation import EvaluationConfig
from tfmplayground.configs.models import ModdedNanoTabPFNClassifierConfig
from tfmplayground.configs.priors import NanoTabICLClassificationPriorConfig
from tfmplayground.configs.training import (
    ClassificationExperimentConfig,
    ClassificationTrainingConfig,
)
from tfmplayground.models.moddednanotabpfn import ModdedNanoTabPFNModel
from tfmplayground.priors import NanoTabICLPrior
from tfmplayground.training.pretrain import pretrainTFM
from tfmplayground.utils import get_default_device, set_randomness_seed

device = get_default_device()
set_randomness_seed(11)  # important for making model weight initialization reproducible

model_config = ModdedNanoTabPFNClassifierConfig(l=10, o=10)
prior_config = NanoTabICLClassificationPriorConfig(max_num_datapoints=2048, min_num_features=5, max_num_features=32)
eval_config = EvaluationConfig(tasks="tabarena", max_n_samples=50000, max_n_features=200)
train_config = ClassificationTrainingConfig(seed=11, batch_size=2, epochs=100, steps=100)
experiment_config = ClassificationExperimentConfig(name="classification_example")

model = pretrainTFM(
    problem="classification",
    model=ModdedNanoTabPFNModel(config=model_config),
    prior=NanoTabICLPrior(config=prior_config, device=device),
    eval=eval_config,
    training=train_config,
    experiment=experiment_config,
)
