from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    """
    settings that evaluation processes use
    """

    tasks: str | list = "toy"
    max_n_features: int = 100
    max_n_samples: int = 1000
    tabarena_dir: str = "tabarena"
    tabarena_method: str = "TFMPlayground"
    tabarena_subset: str | list | None = None
    plot_only: tuple[str, ...] | None = (
        "Linear",
        "RandomForest",
        "CatBoost",
        "RealMLP",
        "TabPFNv2",
        "TabPFN-3",
        "TabICLv2",
        "TabFM",
    )
