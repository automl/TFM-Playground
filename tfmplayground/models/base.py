from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class TabularFoundationModel(nn.Module, ABC):
    """
    base class for every model this package trains

    Attributes
    ----------
    config : dataclass
        config the adapter was built from, set before `nn.Module.__init__` runs
    """

    config: Any

    @abstractmethod
    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """
        predicts test rows with train rows as context
        """
        ...
