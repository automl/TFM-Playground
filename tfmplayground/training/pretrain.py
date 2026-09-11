import torch
from torch import nn

from tfmplayground.configs.evaluation import EvaluationConfig
from tfmplayground.configs.training import (
    ClassificationExperimentConfig,
    ClassificationTrainingConfig,
    ExperimentConfig,
    RegressionExperimentConfig,
    RegressionTrainingConfig,
    TrainingConfig,
)
from tfmplayground.models.base import TabularFoundationModel
from tfmplayground.priors.base import Prior
from tfmplayground.training.callbacks import (
    ClassifierExperimentEvaluationCallback,
    ExperimentEvaluationCallback,
    RegressorExperimentEvaluationCallback,
)
from tfmplayground.training.train import train
from tfmplayground.utils import (
    Experiment,
    get_default_device,
    make_bucket_borders,
    make_regression_decoder,
    set_randomness_seed,
)


def default_training(problem: str) -> TrainingConfig:
    """
    gives training config that matches problem
    """
    if problem == "classification":
        return ClassificationTrainingConfig()
    if problem == "regression":
        return RegressionTrainingConfig()


def default_experiment(problem: str) -> ExperimentConfig:
    """
    gives experiment config that matches problem
    """
    if problem == "classification":
        return ClassificationExperimentConfig()
    if problem == "regression":
        return RegressionExperimentConfig()


def default_callback(
    problem: str,
    experiment: Experiment,
    config: EvaluationConfig,
    device: str | torch.device,
) -> ExperimentEvaluationCallback:
    """
    gives evaluation callback that matches problem
    """
    if problem == "classification":
        return ClassifierExperimentEvaluationCallback(experiment, config=config, device=device)
    if problem == "regression":
        return RegressorExperimentEvaluationCallback(experiment, config=config, device=device)


def check_problems(
    problem: str,
    model: TabularFoundationModel,
    prior: Prior,
    training: TrainingConfig | None,
    experiment: ExperimentConfig | None,
) -> None:
    """
    checks that problem is known and every config agrees with it
    """
    if problem not in ("classification", "regression"):
        raise ValueError(f"{problem!r} problem is not in (classification, regression)")
    for x in (model, prior, training, experiment):
        x_config = getattr(x, "config", x)
        x_problem = getattr(x_config, "problem", None)
        if x_problem is not None and x_problem != problem:
            raise ValueError(f"{type(x).__name__} must do {problem!r}, not {x_problem!r}")


def default_criterion(
    problem: str,
    model: TabularFoundationModel,
    prior: Prior,
    training: TrainingConfig,
    device: str | torch.device,
) -> nn.Module:
    """
    gives loss that matches problem

    fits bucket borders from prior when regression head needs them
    """
    if problem == "classification":
        return nn.CrossEntropyLoss()
    if problem == "regression":
        head = training.criterion if training.criterion is not None else model.config.head
        num_outputs = model.borders.numel() - 1
        if head == "scalar" and num_outputs != 1:
            raise ValueError(f"{head!r} head needs 1 output, not {num_outputs}")
        if head == "buckets":
            if num_outputs < 2:
                raise ValueError(f"{head!r} head needs more than 1 output, not {num_outputs}")
            model.borders = make_bucket_borders(
                prior=prior,
                num_buckets=num_outputs,
                batch_size=training.batch_size,
                min_targets=training.bucket_borders_min_targets,
            ).to(device)
        model.config.head = head
        return make_regression_decoder(model).to(device)


def pretrainTFM(
    problem: str,
    model: TabularFoundationModel,
    prior: Prior,
    eval: EvaluationConfig | None = None,
    training: TrainingConfig | None = None,
    experiment: ExperimentConfig | None = None,
    device: str | torch.device | None = None,
) -> TabularFoundationModel:
    """
    pretrains model on prior and gives it back trained

    Parameters
    ----------
    problem : str
        classification or regression
    model : TabularFoundationModel
        model to pretrain
    prior : Prior
        prior that gives training tables
    eval : EvaluationConfig, optional
        tasks and their limits for periodic evaluations
    training : TrainingConfig, optional
        seed, sizes and limits for training loop
    experiment : ExperimentConfig, optional
        name and directory for experiment artifact tracking
    device : torch.device, optional
        device that holds model and training batches

    Returns
    -------
    TabularFoundationModel
        pretrained model ready for inference
    """
    check_problems(problem, model, prior, training, experiment)
    experimentconfig = experiment if experiment is not None else default_experiment(problem)
    experiment = Experiment(config=experimentconfig)
    training = training if training is not None else default_training(problem)
    set_randomness_seed(training.seed)
    device = device if device is not None else get_default_device()
    eval = eval if eval is not None else EvaluationConfig()
    criterion = default_criterion(problem, model, prior, training, device)
    callback = default_callback(problem, experiment, eval, device)
    experiment.log_configs(model=model.config, prior=prior.config, eval=eval, training=training)

    trained_model = train(
        problem=problem,
        model=model,
        prior=prior,
        criterion=criterion,
        epochs=training.epochs,
        batch_size=training.batch_size,
        steps_per_epoch=training.steps,
        lr=training.lr,
        grad_clip=training.grad_clip,
        device=device,
        callbacks=[callback],
        experiment=experiment,
    )
    return trained_model
