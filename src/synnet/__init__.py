import copy
import logging
import os
import sys
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Optional

import yaml

RUNNING_ON_HPC: bool = "SLURM_JOB_ID" in os.environ

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
    # handlers=[logging.FileHandler(".log"),logging.StreamHandler()],
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def get_loggers(name: str = "synnet"):
    """Get all loggers that contain `name`."""
    return [logging.getLogger(_name) for _name in logging.root.manager.loggerDict if name in _name]


def show_config(args: Namespace):
    """Dump the config to stdout in yaml format."""
    if not getattr(args, "show_config", False):
        return args
    kwargs = vars(args)
    kwargs.pop("show_config")
    f = yaml.dump(kwargs, default_flow_style=False, sort_keys=False, indent=2)
    sys.stdout.write(f)
    exit(0)


def from_config(args: Namespace, sys_args: Optional[Namespace] = None):
    """Get arguments. Priority: user-provided  > config > default."""
    if (config_file := getattr(args, "config", None)) is None:
        return args
    if not Path(config_file).exists():
        raise FileNotFoundError(f"Config file {config_file} not found.")

    # Load config
    with open(Path(config_file), "r") as f:
        config = yaml.safe_load(f)

    # Override: config > default
    kwargs = copy.deepcopy(args).__dict__
    for k, v in config.items():
        if k in kwargs:
            if kwargs[k] != v:
                logger.debug(f"Overriding default: {k}={kwargs[k]} -> {v}")
            kwargs[k] = v
        else:
            warnings.warn(f"Key {k} from config file not found in arguments and ignored.")

    # Override: user-provided > config
    # Get user-provided args
    if sys_args is None:
        sys_args = sys.argv[1:]
    else:
        sys_args = list(sys_args)
    user_keys = [
        k for k in sys_args if k.startswith("-") and not (k == "--config" or k == "--show-config")
    ]  # not very robust
    user_keys = [k.split("=")[0] for k in user_keys]  # wandb sweeps calls with --key=value

    # Update kwargs
    for _k in user_keys:
        k = _k.lstrip("-").replace("-", "_")
        logger.debug(f"Overriding config: {k}={kwargs[k]} -> {getattr(args, k)}")
        kwargs[k] = getattr(args, k)

    # Show new keys from argparse
    _new_keys = [k for k in vars(args) if k not in config if not k == "show_config"]
    if _new_keys:
        warnings.warn(
            f"Arguments `{', '.join(_new_keys)}` not found in config. Is the config file outdated?"
        )
    return Namespace(**kwargs)


def parse_args(args: Namespace, return_dict: bool = True):
    """Helper function to add functionality."""
    args = show_config(args)  # can default config and exits
    args = from_config(args)  # can override args with config file / user
    return args if not return_dict else vars(args)
