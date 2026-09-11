import time

import schedulefree
import torch
from torch import nn

from tfmplayground.models import TabularFoundationModel
from tfmplayground.priors import Prior, PriorDataLoader
from tfmplayground.training.callbacks import Callback
from tfmplayground.utils import (
    Experiment,
    FullSupportBarDistribution,
    QuantileLoss,
    ScalarMSELoss,
    get_default_device,
)


def train(
    problem: str,
    model: TabularFoundationModel,
    prior: Prior,
    criterion: nn.CrossEntropyLoss | ScalarMSELoss | FullSupportBarDistribution | QuantileLoss,
    epochs: int,
    batch_size: int,
    steps_per_epoch: int,
    lr: float,
    grad_clip: float,
    device: str | torch.device | None = None,
    callbacks: list[Callback] | None = None,
    experiment: Experiment | None = None,
) -> TabularFoundationModel:
    """
    trains model on prior batches for given epochs

    steps with non-finite values are skipped, and an epoch where every step is
    skipped raises

    Parameters
    ----------
    problem : str
        classification or regression
    model : TabularFoundationModel
        model to train
    prior : Prior
        prior that gives training tables
    criterion : nn.CrossEntropyLoss or ScalarMSELoss or FullSupportBarDistribution or QuantileLoss
        loss criterion
    epochs : int
        number of epochs to train for
    batch_size : int
        number of tables per batch
    steps_per_epoch : int
        number of batches that make up one epoch
    lr : float
        learning rate for optimizer
    grad_clip : float
        largest gradient norm to keep before each step
    device : torch.device, optional
        device that holds model and batches
    callbacks : list of Callback, optional
        callbacks to run at end of each epoch, and to close at end of run
    experiment : Experiment, optional
        experiment that saves checkpoints after each epoch

    Returns
    -------
    TabularFoundationModel
        model after last epoch
    """
    if callbacks is None:
        callbacks = []
    if not device:
        device = get_default_device()
    model.to(device)
    criterion = criterion.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    classification_task = problem == "classification"
    regression_task = not classification_task
    batches = iter(PriorDataLoader(prior, batch_size))

    try:
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            model.train()  # Turn on the train mode
            optimizer.train()
            total_loss = 0.0
            num_valid = 0
            for _ in range(steps_per_epoch):
                x_train, y_train, x_test, y_test = next(batches)
                x_train = x_train.to(device)
                y_train = y_train.to(device)
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                if not all(torch.isfinite(t).all() for t in (x_train, y_train, x_test, y_test)):
                    continue

                if regression_task:
                    y_mean = y_train.mean(dim=1, keepdim=True)
                    y_std = y_train.std(dim=1, keepdim=True) + 1e-8
                    y_train = (y_train - y_mean) / y_std
                    y_test = (y_test - y_mean) / y_std
                    if not torch.isfinite(y_train).all() or not torch.isfinite(y_test).all():
                        continue

                output = model(x_train, y_train, x_test)
                if not torch.isfinite(output).all():
                    continue
                if classification_task:
                    y_test = y_test.reshape((-1,)).to(torch.long)
                    output = output.reshape(-1, output.shape[-1])
                    output_classes = output.shape[-1]
                    prior_classes = int(y_test.max()) + 1
                    if prior_classes > output_classes:
                        raise ValueError(f"model has {output_classes} outputs, prior gives {prior_classes} classes")

                losses = criterion(output, y_test)
                if not torch.isfinite(losses).all():
                    continue

                loss = losses.mean()
                loss.backward()
                total_loss += loss.cpu().detach().item()
                num_valid += 1

                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            if num_valid == 0:
                raise RuntimeError(f"all {steps_per_epoch} steps in epoch {epoch} had non-finite values")

            end_time = time.time()
            mean_loss = total_loss / num_valid
            model.eval()
            optimizer.eval()

            for callback in callbacks:
                callback.on_epoch_end(epoch, end_time - epoch_start_time, mean_loss, model)

            if experiment is not None:
                experiment.save_checkpoints(model)
    except KeyboardInterrupt:
        pass
    finally:
        for callback in callbacks:
            callback.close()

    return model
