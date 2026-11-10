from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    """
    settings that evaluation processes use

    suite names a set of openml tasks: "toy", "tabarena", or an openml suite name. task_ids lists openml
    task ids directly and, when given, is used instead of suite
    """

    suite: str | None = "toy"
    task_ids: list[int] | None = None
    max_n_features: int = 100
    max_n_samples: int = 1000
