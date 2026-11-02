from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    """
    settings that evaluation processes use
    """

    tasks: str | list = "toy"
    max_n_features: int = 100
    max_n_samples: int = 1000
    tabarena_subset: str | list | None = None
    tabarena_dir: str = "tabarena"
    plot_only: tuple[str, ...] | None = (
        "Linear",
        "RandomForest",
        "CatBoost",
        "RealMLP",
        "TabICLv2",
        "TabFM",
        "TabPFNv2",
        "TabPFN-3",
    )
