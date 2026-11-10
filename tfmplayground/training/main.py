"""
pretrains from a yaml run config, with hydra's compose api instead of its decorator

    python -m tfmplayground.training --config-name classification
    python -m tfmplayground.training --config-name regression model=tabfm_regressor training.epochs=2
    python -m tfmplayground.training --config-name quickstart experiment.name=my_run

anything argparse does not know is passed on to hydra as an override: `group=option` swaps a yaml file in
`<config-dir>/<group>/`, `group.field=value` sets one field. unknown fields are rejected, because every yaml
file refines one of the dataclasses in `tfmplayground.configs`
"""

import argparse
from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tfmplayground.configs.store import RunConfig
from tfmplayground.training.pretrain import pretrain_from_config


def main() -> None:
    """
    composes the run config from the command line and pretrains
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="classification", help="yaml file in the config dir to start from")
    parser.add_argument("--config-dir", default="yaml_configs", help="directory holding the yaml tree")
    known, overrides = parser.parse_known_args()

    config_dir = Path(known.config_dir).resolve()
    if not config_dir.is_dir():
        raise SystemExit(f"config dir {config_dir} does not exist, run from the repo root or pass --config-dir")
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name=known.config_name, overrides=overrides)
    print(OmegaConf.to_yaml(cfg))
    pretrain_from_config(cast(RunConfig, OmegaConf.to_object(cfg)))
