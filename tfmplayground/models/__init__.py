from tfmplayground.configs import models as configs
from tfmplayground.models.base import TabularFoundationModel
from tfmplayground.models.moddednanotabpfn import ModdedNanoTabPFNModel
from tfmplayground.models.nanotabicl import NanoTabICLModel
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.models.tabfm import TabFMModel
from tfmplayground.models.tabicl import TabICLModel

# which adapter a config dataclass belongs to, by the base config its classifier and regressor variants share;
# a list of pairs rather than a dict, so the lookup order is explicit and editors do not mistake the classes
# for unhashable dataclass instances
MODEL_FOR_CONFIG: list[tuple[type, type[TabularFoundationModel]]] = [
    (configs.NanoTabPFNModelConfig, NanoTabPFNModel),
    (configs.ModdedNanoTabPFNModelConfig, ModdedNanoTabPFNModel),
    (configs.NanoTabICLModelConfig, NanoTabICLModel),
    (configs.TabICLModelConfig, TabICLModel),
    (configs.TabFMModelConfig, TabFMModel),
]


def build_model(config) -> TabularFoundationModel:
    """
    builds the model adapter whose config dataclass this is

    Parameters
    ----------
    config : dataclass
        one of the classifier or regressor configs in `tfmplayground.configs.models`

    Returns
    -------
    TabularFoundationModel
        freshly initialized model holding config
    """
    for config_class, model_class in MODEL_FOR_CONFIG:
        if isinstance(config, config_class):
            return model_class(config=config)
    known = ", ".join(config_class.__name__ for config_class, _ in MODEL_FOR_CONFIG)
    raise TypeError(f"no model takes {type(config).__name__}, known configs are {known}")


__all__ = [
    "MODEL_FOR_CONFIG",
    "TabularFoundationModel",
    "ModdedNanoTabPFNModel",
    "NanoTabICLModel",
    "NanoTabPFNModel",
    "TabFMModel",
    "TabICLModel",
    "build_model",
]
