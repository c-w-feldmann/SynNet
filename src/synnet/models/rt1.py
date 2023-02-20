"""Reactant1 network (for predicting 1st reactant).
"""
import logging
from typing import Callable

import pytorch_lightning as pl
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar

import synnet
from synnet import RUNNING_ON_HPC
from synnet.models.common import _compute_class_weights_from_dataloader, init_save_dir
from synnet.models.mlp import MLP
from synnet.utils.data import (
    get_dataloaders,
    get_dataset,
    get_datasets_act,
    get_datasets_rt1,
    get_datasets_rt2,
    get_datasets_rxn,
    get_splits,
)

logger = logging.getLogger(__name__)

MAX_PROCESSES = 8
GET_DATATSET_MAP: dict[str, Callable] = {
    "act": get_datasets_act,
    "rt1": get_datasets_rt1,
    "rt2": get_datasets_rt2,
    "rxn": get_datasets_rxn,
}


def get_args():
    import argparse

    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--project", default="synnet", type=str)
    parser.add_argument("--description", default=None, type=str)
    parser.add_argument("--group", default="rt1", type=str)
    parser.add_argument("--result_dir", default="results/", type=str)

    # config
    parser.add_argument("--show-config", action="store_true", default=False)
    parser.add_argument("--config", type=str, default=None)

    # Sweep?
    parser.add_argument("--sweep_config", type=str)

    # launcher_args
    parser.add_argument("--script_name", default=__file__, type=str)
    parser.add_argument("--slurm_script", default="slurm.sh", type=str)
    parser.add_argument("--use_slurm", default="false", type=str)
    parser.add_argument("--visible_devices", default=[0], type=list)
    parser.add_argument("--ncpu", default=MAX_PROCESSES, type=int)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--fast_dev_run", default=False, action="store_true")

    # train
    parser.add_argument("--max_epochs", default=601, type=int)
    parser.add_argument("--min_epochs", default=1, type=int)
    parser.add_argument("--early-stopping-patience", default=3, type=int)

    # data
    parser.add_argument(
        "--data", default="data-st/final/datasets/syntrees_iclr_like_dev.json.gz", type=str
    )
    ## state
    parser.add_argument("--embedding_state_nbits", default=4096, type=int)
    parser.add_argument("--embedding_state_radius", default=2, type=int)
    ## rt1
    parser.add_argument("--embedding_rct_nbits", default=256, type=int)
    parser.add_argument("--embedding_rct_radius", default=2, type=int)

    # parameters
    parser.add_argument("--task", default="regression", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--input_dim", default=12288, type=int)
    parser.add_argument("--output_dim", default=256, type=int)
    parser.add_argument("--hidden_dim", default=3800, type=int)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--num_dropout_layers", default=1, type=int)
    parser.add_argument("--dropout", default=0.4, type=float)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--learning_rate", default=3e-05, type=float)
    parser.add_argument("--loss", default="cosine_distance", type=str)
    parser.add_argument("--valid_loss", default="cosine_distance", type=str)
    parser.add_argument("--val_freq", default=1, type=int)
    parser.add_argument("--with_class_weights", default=False, action="store_true")

    return parser.parse_args()


def train():
    """Train network"""

    # Set up logging dir
    save_dir = init_save_dir(
        kwargs["result_dir"],
        suffix=kwargs.get("group", "") + ("_debug" if kwargs["debug"] else ""),
    )
    kwargs["save_dir"] = str(save_dir)

    # Dump args
    with open(save_dir / "kwargs.yaml", "wt") as f:
        yaml.dump(kwargs, f, indent=2, default_flow_style=False, sort_keys=False)

    # Set up loggers
    wandb_logger = pl_loggers.WandbLogger(
        save_dir=save_dir,
        name=kwargs["name"],
        project=kwargs["project"],
        group=kwargs["group"] + ("-debug" if kwargs["debug"] else ""),
    )
    wandb_logger.experiment.config.update({"kwargs": kwargs})

    csv_logger = pl_loggers.CSVLogger(save_dir, name="")
    loggers = [csv_logger, wandb_logger]
    logger.info(f"Save dir: `{save_dir}`, CSV log dir: `{csv_logger.log_dir}`")

    pl.seed_everything(0)

    # region Data
    # Set up dataloaders
    dataset = get_dataset(kwargs["data"])
    train, valid, _ = get_splits(dataset)

    # Get dataset
    get_dataset_fct = GET_DATATSET_MAP[kwargs["group"]]
    datasets = get_dataset_fct(train, valid, None, num_workers=kwargs["ncpu"], **kwargs)

    train_loader, val_loader, _ = get_dataloaders(
        *datasets,
        batch_size=kwargs["batch_size"],
        num_workers=kwargs["ncpu"],
        verbose=True,
    )
    logger.info(f"Set up dataloaders.")

    # Compute class weights?
    if kwargs["with_class_weights"]:
        class_weights = _compute_class_weights_from_dataloader(train_loader)
        kwargs["class_weights"] = class_weights

    # endregion

    # Model
    mlp = MLP(**kwargs)

    # Callbacks
    callbacks = {
        "modelcheckpoint": ModelCheckpoint(
            monitor="val_loss",
            dirpath=save_dir / "checkpoints",
            filename="{epoch}-{val_loss:.3f}",
        ),
        "tqdm": TQDMProgressBar(
            refresh_rate=max(1, int(len(train_loader) * 0.05)),
        ),
        "earlystopping": EarlyStopping(
            monitor="val_loss",
            patience=kwargs["early_stopping_patience"],
        ),
    }
    if RUNNING_ON_HPC:
        callbacks.pop("tqdm")  # dont spam the logs

    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=kwargs["visible_devices"],
        max_epochs=kwargs["max_epochs"],
        min_epochs=kwargs["min_epochs"],
        callbacks=list(callbacks.values()),
        logger=loggers,
        check_val_every_n_epoch=kwargs["val_freq"],
        fast_dev_run=kwargs["fast_dev_run"],
    )

    logger.info(f"Start training")
    trainer.fit(mlp, train_loader, val_loader)
    logger.info(f"Training completed.")
    logger.info(f"Save dir: {save_dir}, CSV log dir: {csv_logger.log_dir}")
    return trainer


if __name__ == "__main__":
    # Parse input args
    args = get_args()
    kwargs = synnet.parse_args(args, return_dict=True)
    logger.info(f"Arguments:\n{yaml.dump(kwargs,indent=2)}")

    trainer = train()
