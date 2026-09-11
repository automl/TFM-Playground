from abc import ABC, abstractmethod
from datetime import datetime

import torch
from sklearn.metrics import r2_score, roc_auc_score

from tfmplayground.configs.evaluation import EvaluationConfig
from tfmplayground.evaluation.evaluation import get_openml_predictions
from tfmplayground.interface import TabularClassifier, TabularRegressor
from tfmplayground.models.base import TabularFoundationModel
from tfmplayground.utils import Experiment


class Callback(ABC):
    """Abstract base class for callbacks."""

    @abstractmethod
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        """
        Called at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            epoch_time (float): Time of the epoch in seconds.
            loss (float): Mean loss for the epoch.
            model: The model being trained.
            **kwargs: Additional arguments.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Called to release any resources or perform cleanup.
        """
        pass


class BaseLoggerCallback(Callback):
    """Abstract base class for logger callbacks."""

    pass


class ConsoleLoggerCallback(BaseLoggerCallback):
    """Logger callback that prints epoch information to the console."""

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        print(f"Epoch {epoch:5d} | Time {epoch_time:5.2f}s | Mean Loss {loss:5.2f}", flush=True)

    def close(self):
        """Nothing to clean up for print logger."""
        pass


class TensorboardLoggerCallback(BaseLoggerCallback):
    """Logger callback that logs epoch information to TensorBoard."""

    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir)

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        self.writer.add_scalar("Loss/train", loss, epoch)
        self.writer.add_scalar("Time/epoch", epoch_time, epoch)

    def close(self):
        self.writer.close()


class WandbLoggerCallback(BaseLoggerCallback):
    """Logger callback that logs epoch information to Weights & Biases."""

    def __init__(self, project: str, name: str = None, config: dict = None, log_dir: str = None):
        """
        Initializes a WandbLoggerCallback.

        Args:
            project (str): The name of the wandb project.
            name (str, optional): The name of the run. Defaults to None.
            config (dict, optional): Configuration dictionary for the run. Defaults to None.
            log_dir (str, optional): Directory to save wandb logs. Defaults to None.
        """
        try:
            import wandb

            self.wandb = wandb  # store wandb module to avoid import if not used
            wandb.init(project=project, name=name, id=name, config=config, dir=log_dir, resume="allow")
        except ImportError as e:
            raise ImportError("wandb is not installed. Install it with: pip install wandb") from e

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        log_dict = {"epoch": epoch, "loss": loss, " epoch_time": epoch_time}
        self.wandb.log(log_dict)

    def close(self):
        self.wandb.finish()


class ExperimentCallback(BaseLoggerCallback):
    """
    keeps experiment informed of training progress
    """

    def __init__(self, experiment: Experiment) -> None:
        """
        starts experiment log for this run
        """
        self.experiment = experiment
        self.experiment.log(f"experiment: {self.experiment.id}", console=True)

    def on_epoch_end(
        self,
        epoch: int,
        epoch_time: float,
        loss: float,
        model: TabularFoundationModel,
        **kwargs,
    ) -> None:
        """
        records progress of one epoch
        """
        self.experiment.log(f"epoch {epoch} | epoch time {epoch_time:.2f}s | mean loss {loss:.2f}")

    def close(self) -> None:
        """
        closes run with time it took
        """
        minutes = (datetime.now() - self.experiment.started).total_seconds() / 60
        self.experiment.log(f"runtime: {minutes:.2f} mins")


class ExperimentEvaluationCallback(ExperimentCallback):
    """
    measures how good model is while it trains
    """

    def __init__(
        self,
        experiment: Experiment,
        config: EvaluationConfig,
        device: str | torch.device,
    ) -> None:
        """
        takes what evaluation needs
        """
        super().__init__(experiment)
        self.config = config
        self.device = device

    def predictions(self, model: TabularClassifier | TabularRegressor) -> dict[str, tuple]:
        """
        asks model to predict tasks chosen for evaluation
        """
        return get_openml_predictions(
            model=model,
            tasks=self.config.tasks,
            max_n_features=self.config.max_n_features,
            max_n_samples=self.config.max_n_samples,
        )

    def evaluate(self, model: TabularFoundationModel, **kwargs) -> list[float]:
        """
        leaves scoring to subclass, because metric follows problem
        """
        raise NotImplementedError

    def on_epoch_end(
        self,
        epoch: int,
        epoch_time: float,
        loss: float,
        model: TabularFoundationModel,
        **kwargs,
    ) -> None:
        """
        scores model and records how good this epoch was
        """
        scores = self.evaluate(model, **kwargs)
        if not scores:
            raise ValueError("scores are empty, nothing to average")
        mean = sum(scores) / len(scores)
        self.experiment.score = mean
        fields = [
            f"epoch {epoch}",
            f"epoch time {epoch_time:.2f}s",
            f"mean loss {loss:.2f}",
            f"{self.metric} {mean:.2f}",
            f"tasks {len(scores)}",
        ]
        line = " | ".join(fields)
        self.experiment.log(line, console=True)


class ClassifierExperimentEvaluationCallback(ExperimentEvaluationCallback):
    """
    measures classifier quality with roc auc
    """

    problem = "classification"
    metric = "mean roc auc"

    def evaluate(self, model: TabularFoundationModel, **kwargs) -> list[float]:
        """
        scores model as classifier on every task
        """
        classifier = TabularClassifier(model, self.device)
        predictions = self.predictions(classifier)
        scores = [roc_auc_score(y_true, y_proba, multi_class="ovr") for y_true, _, y_proba in predictions.values()]
        return scores


class RegressorExperimentEvaluationCallback(ExperimentEvaluationCallback):
    """
    measures regressor quality with r2
    """

    problem = "regression"
    metric = "mean r2"

    def evaluate(self, model: TabularFoundationModel, **kwargs) -> list[float]:
        """
        scores model as regressor on every task
        """
        regressor = TabularRegressor(model, device=self.device)
        predictions = self.predictions(regressor)
        scores = [r2_score(y_true, y_pred) for y_true, y_pred, _ in predictions.values()]
        return scores
