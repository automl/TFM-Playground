from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TrainingConfig(ABC):
    """
    settings training runs share
    """

    seed: int = 2402
    lr: float = 1e-4
    batch_size: int = 1
    steps: int = 100
    epochs: int = 10000
    grad_clip: float = 1.0

    @property
    @abstractmethod
    def problem(self) -> str: ...


@dataclass
class ClassificationTrainingConfig(TrainingConfig):
    """
    training settings for classification
    """

    problem: str = "classification"


@dataclass
class RegressionTrainingConfig(TrainingConfig):
    """
    training settings for regression
    """

    problem: str = "regression"
    criterion: str | None = None
    bucket_borders_min_targets: int = 1_000_000


@dataclass
class ExperimentConfig(ABC):
    """
    settings experiment tracking shares
    """

    name: str = "test"
    experiments_dir: str = "workdir/experiments"

    @property
    @abstractmethod
    def problem(self) -> str: ...


@dataclass
class ClassificationExperimentConfig(ExperimentConfig):
    """
    experiment settings for classification
    """

    problem: str = "classification"


@dataclass
class RegressionExperimentConfig(ExperimentConfig):
    """
    experiment settings for regression
    """

    problem: str = "regression"
