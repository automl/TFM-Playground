"""
hydra config store for composing runs from yaml

the yaml tree in `yaml_configs/` only ever overrides fields of the dataclasses in this package. every yaml
file starts its defaults list with the name of the dataclass it refines, and this module registers each
dataclass under that name, so `OmegaConf.to_object()` gives dataclass instances back instead of dicts.

`compose()` itself is called by the entry point that wants a config, see `tfmplayground/training/main.py`.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from tfmplayground.configs import models as model_configs
from tfmplayground.configs import priors as prior_configs
from tfmplayground.configs.evaluation import EvaluationConfig
from tfmplayground.configs.training import (
    ClassificationExperimentConfig,
    ClassificationTrainingConfig,
    ExperimentConfig,
    RegressionExperimentConfig,
    RegressionTrainingConfig,
    TrainingConfig,
)


@dataclass
class RunConfig:
    """
    everything one pretraining run is configured with

    model and prior are typed Any because any of the model or prior config dataclasses may sit there;
    `OmegaConf.to_object()` still gives back the concrete dataclass each yaml names in its defaults
    """

    problem: str = MISSING
    model: Any = MISSING
    prior: Any = MISSING
    training: TrainingConfig = MISSING
    experiment: ExperimentConfig = MISSING
    eval: EvaluationConfig = field(default_factory=EvaluationConfig)
    device: str | None = None


def _public_dataclasses(module) -> list[type]:
    """
    gives dataclasses that module defines itself, in definition order
    """
    return [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__ and hasattr(cls, "__dataclass_fields__")
    ]


def register_configs() -> ConfigStore:
    """
    registers every config dataclass under its class name in its hydra group

    groups are `model`, `prior`, `training`, `experiment` and `eval`, matching the top-level fields of
    `RunConfig`, and `RunConfig` itself is registered without a group. the abstract training and
    experiment base configs are left out since yaml cannot refine them into something instantiable.
    registering twice is harmless
    """
    cs = ConfigStore.instance()
    cs.store(name="RunConfig", node=RunConfig)
    for cls in _public_dataclasses(model_configs):
        cs.store(group="model", name=cls.__name__, node=cls)
    for cls in _public_dataclasses(prior_configs):
        cs.store(group="prior", name=cls.__name__, node=cls)
    for cls in (ClassificationTrainingConfig, RegressionTrainingConfig):
        cs.store(group="training", name=cls.__name__, node=cls)
    for cls in (ClassificationExperimentConfig, RegressionExperimentConfig):
        cs.store(group="experiment", name=cls.__name__, node=cls)
    cs.store(group="eval", name=EvaluationConfig.__name__, node=EvaluationConfig)
    return cs


register_configs()
