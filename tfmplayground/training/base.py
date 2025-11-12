from abc import ABC, abstractmethod

import torch.nn as nn

class Callback(ABC):
    """ Abstract base class for callbacks."""

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


class Trainer(ABC):
    """Trainer class for training models."""

    @abstractmethod
    def train(self) -> nn.Module:
        """
        Trains the given model on the provided dataset.

        Args:
            model: The model to be trained.
            train_dataset: The dataset to train the model on.
            callbacks (list[Callback]): List of callback instances to be used during training.
            run_dir (str): Directory for saving training outputs.
            run_name (str): Name of the training run.

        Returns:
            The trained model and final loss.
        """
        pass