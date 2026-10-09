from abc import ABC, abstractmethod
from collections.abc import Iterator

import torch


class Prior(ABC):
    """
    base class for every prior this package trains on
    """

    config = None

    @abstractmethod
    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        gives one batch of tables, split into train and test parts
        """
        ...


class PriorDataLoader:
    """
    turns prior into endless iterator of batches
    """

    def __init__(self, prior: Prior, batch_size: int) -> None:
        """
        keeps prior and batch size
        """
        self.prior = prior
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        gives batches from prior without end
        """
        while True:
            yield self.prior.batch(self.batch_size)
