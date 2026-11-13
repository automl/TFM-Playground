from collections.abc import Callable

import torch

from tfmplayground.configs import priors as configs

from .base import Prior, PriorDataLoader
from .dump import DumpPrior, PriorDumpDataLoader
from .nanotabicl import NanoTabICLPrior
from .tabicl import TabICLPrior, TabICLPriorDataLoader
from .tabpfn import TabPFNPriorDataLoader, build_tabpfn_prior
from .ticl import TICLPriorDataLoader, build_ticl_prior

# which prior a config dataclass belongs to, by the base config its classification and regression variants share;
# a list of pairs like MODEL_FOR_CONFIG, with callables since the Prior ABC declares no __init__
PRIOR_FOR_CONFIG: list[tuple[type, Callable[..., Prior]]] = [
    (configs.NanoTabICLPriorConfig, NanoTabICLPrior),
    (configs.TabICLPriorConfig, TabICLPrior),
    (configs.PriorDumpConfig, DumpPrior),
]


def build_prior(config, device: str | torch.device | None = None) -> Prior:
    """
    builds the prior whose config dataclass this is

    Parameters
    ----------
    config : dataclass
        one of the classification or regression configs in `tfmplayground.configs.priors`
    device : torch.device, optional
        device that batches are emitted on

    Returns
    -------
    Prior
        prior holding config
    """
    for config_class, prior_class in PRIOR_FOR_CONFIG:
        if isinstance(config, config_class):
            return prior_class(config=config, device=device)
    known = ", ".join(config_class.__name__ for config_class, _ in PRIOR_FOR_CONFIG)
    raise TypeError(f"no prior takes {type(config).__name__}, known configs are {known}")


__all__ = [
    "PRIOR_FOR_CONFIG",
    "Prior",
    "PriorDataLoader",
    "DumpPrior",
    "PriorDumpDataLoader",
    "NanoTabICLPrior",
    "TabICLPrior",
    "TabICLPriorDataLoader",
    "TICLPriorDataLoader",
    "TabPFNPriorDataLoader",
    "build_prior",
    "build_ticl_prior",
    "build_tabpfn_prior",
]
