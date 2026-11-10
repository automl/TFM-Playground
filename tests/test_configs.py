from pathlib import Path
from typing import cast

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

import tfmplayground.configs.store  # noqa: F401  registers the schemas the yaml files refine
from tfmplayground.configs.evaluation import EvaluationConfig
from tfmplayground.configs.models import NanoTabPFNClassifierConfig
from tfmplayground.configs.store import RunConfig
from tfmplayground.configs.training import ClassificationTrainingConfig
from tfmplayground.evaluation.evaluation import TOY_TASKS, get_task_ids
from tfmplayground.models import build_model
from tfmplayground.priors import NanoTabICLPrior, build_prior

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "yaml_configs"


def _yaml_names(group: str) -> list[str]:
    return sorted(path.stem for path in (CONFIGS_DIR / group).glob("*.yaml"))


def _compose(config_name: str, *overrides: str) -> RunConfig:
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    return cast(RunConfig, OmegaConf.to_object(cfg))


@pytest.mark.parametrize("config_name", ["classification", "regression"])
def test_root_config_gives_run_config_of_dataclasses(config_name):
    cfg = _compose(config_name)
    assert isinstance(cfg, RunConfig)
    assert cfg.problem == config_name
    assert isinstance(cfg.eval, EvaluationConfig)
    for part in (cfg.model, cfg.prior, cfg.training, cfg.experiment):
        assert hasattr(part, "__dataclass_fields__"), f"{part!r} is not a dataclass"
        assert part.problem == config_name


# every group yaml must name a registered dataclass and set only fields it has. the host run config does not
# matter for that: composition never checks `problem`, only `check_problems` does at run time


@pytest.mark.parametrize("model", _yaml_names("model"))
def test_every_model_yaml_composes(model):
    cfg = _compose("classification", f"model={model}")
    assert hasattr(cfg.model, "__dataclass_fields__")


@pytest.mark.parametrize("prior", _yaml_names("prior"))
def test_every_prior_yaml_composes(prior):
    cfg = _compose("classification", f"prior={prior}")
    assert hasattr(cfg.prior, "__dataclass_fields__")


@pytest.mark.parametrize("training", _yaml_names("training"))
def test_every_training_yaml_composes(training):
    cfg = _compose("classification", f"training={training}")
    assert hasattr(cfg.training, "__dataclass_fields__")


def test_debug_composes_small_parts():
    cfg = _compose("debug")
    assert cfg.model.num_layers == 2
    assert cfg.prior.max_num_datapoints == 50
    assert cfg.training.steps == 5
    assert cfg.experiment.name == "debug"


def test_overrides_are_typed():
    cfg = _compose("classification", "model=nanotabpfn_classifier", "model.num_layers=2", "training.lr=0.5")
    assert isinstance(cfg.model, NanoTabPFNClassifierConfig)
    assert cfg.model.num_layers == 2
    assert isinstance(cfg.training, ClassificationTrainingConfig)
    assert cfg.training.lr == 0.5


def test_eval_task_ids_override_suite():
    cfg = _compose("classification", "eval.task_ids=[59,2382]")
    assert cfg.eval.suite == "toy"
    assert cfg.eval.task_ids == [59, 2382]
    assert get_task_ids(cfg.eval.suite, cfg.eval.task_ids) == [59, 2382]
    assert get_task_ids(_compose("classification").eval.suite) == TOY_TASKS


def test_unknown_field_is_rejected():
    with pytest.raises(ConfigCompositionException, match="num_layerz"):
        _compose("classification", "model.num_layerz=2")


def test_quickstart_refines_classification():
    cfg = _compose("quickstart")
    assert isinstance(cfg.model, NanoTabPFNClassifierConfig)
    assert cfg.prior.max_num_datapoints == 50
    assert cfg.training.batch_size == 2
    assert cfg.experiment.name == "quickstart"


def test_build_from_debug_config():
    cfg = _compose("debug")
    model = build_model(cfg.model)
    assert model.config is cfg.model
    prior = build_prior(cfg.prior, device="cpu")
    assert isinstance(prior, NanoTabICLPrior)


def test_saved_run_config_composes_again(tmp_path):
    from tfmplayground.configs.training import ClassificationExperimentConfig
    from tfmplayground.utils import Experiment

    cfg = _compose("debug", "model.num_layers=3", "eval.task_ids=[59]")
    experiment_config = ClassificationExperimentConfig(name="saved", experiments_dir=str(tmp_path))
    run = Experiment(experiment_config)
    run.save_run_config(
        problem=cfg.problem,
        model=cfg.model,
        prior=cfg.prior,
        training=cfg.training,
        experiment=experiment_config,
        eval=cfg.eval,
        device="cpu",
    )
    with initialize_config_dir(config_dir=str(run.dir), version_base="1.3"):
        again = cast(RunConfig, OmegaConf.to_object(compose(config_name=f"{run.id}-config")))
    assert again.model == cfg.model
    assert again.prior == cfg.prior
    assert again.training == cfg.training
    assert again.experiment == experiment_config
    assert again.eval == cfg.eval
    assert again.problem == "classification"
    assert again.device == "cpu"


def test_build_rejects_foreign_config():
    with pytest.raises(TypeError, match="no model takes EvaluationConfig"):
        build_model(EvaluationConfig())
