from importlib.metadata import version
from pathlib import Path

import pandas as pd
import torch
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.contexts.tabarena.methods import tabarena_method_metadata_collection
from tabarena.utils.config_utils import ConfigGenerator

from tfmplayground.configs.evaluation import EvaluationConfig
from tfmplayground.interface import TabularClassifier, TabularRegressor
from tfmplayground.utils import load_model


class TFMPlaygroundModel(AbstractTorchModel):
    """
    autogluon wrapper, so tabarena can fit and score a playground checkpoint
    """

    default_num_gpus = 1

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_gpus: float = 0, **kwargs) -> None:
        device = self._resolve_fit_device(num_gpus=num_gpus) if torch.cuda.is_available() else "cpu"
        model = load_model(self._get_model_params()["checkpoint"])
        interface = TabularRegressor if self.problem_type == "regression" else TabularClassifier
        self.model = interface(model, device=device).fit(self.preprocess(X), y.to_numpy())

    def _get_default_auxiliary_params(self) -> dict:
        params = super()._get_default_auxiliary_params()
        params.update({"valid_raw_types": ["int", "float", "category"]})
        return params

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass", "regression"]

    def get_device(self) -> str:
        return str(self.model.device)

    def _set_device(self, device: str) -> None:
        self.model.device = device
        self.model.model = self.model.model.to(device)


def tabarena_version() -> str:
    """
    gives tabarena package version and date of its newest baseline, so results say which leaderboard they rank on
    """
    dates = [m.date for m in tabarena_method_metadata_collection.method_metadata_lst if m.date]
    return f"{version('tabarena')} (baselines up to {max(dates)})"


def evaluate_on_tabarena(checkpoint: str | Path, output_dir: str | Path, config: EvaluationConfig) -> pd.Series:
    """
    fits checkpoint on tabarena tasks of its problem, and gives its row of the leaderboard against tabarena baselines

    plot_only keeps only these baselines and ours in figures, leaderboard still counts every baseline
    """
    output_dir = Path(output_dir)
    problem = load_model(checkpoint).config.problem
    subset = config.tabarena_subset
    subset = [subset] if isinstance(subset, str) else list(subset)
    method = config.tabarena_method
    TFMPlaygroundModel.ag_key = TFMPlaygroundModel.ag_name = method
    generator = ConfigGenerator(
        model_cls=TFMPlaygroundModel,
        manual_configs=[{"checkpoint": str(checkpoint)}],
        search_space={},
    )
    experiments = TabArenaV0pt1ExperimentBundle(models=[(generator, 0)]).build_experiments()
    context = TabArenaContext()
    context.build_and_run_jobs(
        experiments,
        expname=str(output_dir / "results"),
        subset=[*subset, problem],
        debug_mode=True,
        raise_on_failure=False,
    )
    plot_only = config.plot_only
    if plot_only is not None:
        plot_only = [*plot_only, method]
    leaderboard = context.compare(output_dir=output_dir / "eval", plot_only=plot_only)
    return leaderboard[leaderboard["method"].str.startswith(method)].iloc[0]
