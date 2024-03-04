"""Reactant1 network (for predicting 1st reactant).
"""

import argparse
import json
import logging
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from synnet import RUNNING_ON_HPC
from synnet.models.common import init_save_dir, xy_to_dataloader
from synnet.models.mlp import MLP

MAX_PROCESSES = 8
logger = logging.getLogger(__file__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--project", default="synnet", type=str)
    parser.add_argument("--description", default="default config", type=str)
    parser.add_argument("--group", default="rt1", type=str)
    parser.add_argument("--result_dir", default="tmp/rt1", type=str)

    # Sweep?
    parser.add_argument("--sweep_config", type=str)

    # launcher_args
    parser.add_argument("--script_name", default="src/synnet/models/rt1.py", type=str)
    parser.add_argument("--slurm_script", default="slurm.sh", type=str)
    parser.add_argument("--use_slurm", default="false", type=str)
    parser.add_argument("--visible_devices", default=[0], type=int, nargs="+")
    parser.add_argument("--ncpu", default=MAX_PROCESSES, type=int)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--fast_dev_run", default=False, action="store_true")

    # data
    parser.add_argument(
        "--Xtrain_file", default="data/featurized-uni/Xy/X_rt1_train.npz", type=str
    )
    parser.add_argument(
        "--ytrain_file", default="data/featurized-uni/Xy/y_rt1_train.npz", type=str
    )
    parser.add_argument(
        "--Xvalid_file", default="data/featurized-uni/Xy/X_rt1_valid.npz", type=str
    )
    parser.add_argument(
        "--yvalid_file", default="data/featurized-uni/Xy/y_rt1_valid.npz", type=str
    )
    parser.add_argument("--aug-fraction", default=0.3, type=float)

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
    parser.add_argument("--max_epochs", default=601, type=int)
    parser.add_argument("--loss", default="cosine_distance", type=str)
    parser.add_argument("--valid_loss", default="cosine_distance", type=str)
    parser.add_argument("--val_freq", default=10, type=int)

    return parser.parse_args()


def train() -> None:
    logger.info("Start.")

    # Parse input args
    args = get_args()
    kwargs = args.__dict__
    logger.info(f"Arguments: {json.dumps(kwargs,indent=2)}")

    result_dir = kwargs.get("result_dir", None)
    if result_dir is None:
        raise ValueError("result_dir must be specified.")
    # Set up logging dir
    save_dir = init_save_dir(result_dir, suffix=("-debug" if kwargs["debug"] else ""))

    # Dump args
    with open(save_dir / "kwargs.yaml", "wt") as f:
        yaml.dump(kwargs, f, indent=2, default_flow_style=False)

    pl.seed_everything(0)

    # Set up dataloaders
    train_dataloader = xy_to_dataloader(
        X_file=Path(kwargs["Xtrain_file"]),
        y_file=Path(kwargs["ytrain_file"]),
        batch_size=kwargs["batch_size"],
        task=kwargs["task"],
        num_workers=kwargs["ncpu"],
        n=None if not kwargs["debug"] else 250,
        shuffle=True,
    )

    valid_dataloader = xy_to_dataloader(
        X_file=Path(kwargs["Xvalid_file"]),
        y_file=Path(kwargs["yvalid_file"]),
        batch_size=kwargs["batch_size"],
        task=kwargs["task"],
        num_workers=kwargs["ncpu"],
        n=None if not kwargs["debug"] else 250,
        shuffle=False,
    )

    logger.info("Set up dataloaders.")

    # Instantiate MLP
    mlp = MLP(**kwargs)

    # Set up Trainer
    csv_logger = pl_loggers.CSVLogger(save_dir, name="")
    log_dir = csv_logger.log_dir
    logger.info(f"Model log dir set to: {log_dir}")

    callbacks = {
        "modelcheckpoint": ModelCheckpoint(
            monitor="val_loss", dirpath=log_dir, filename="ckpts.{epoch}-{val_loss:.3f}"
        ),
        "tqdm": TQDMProgressBar(refresh_rate=max(1, int(len(train_dataloader) * 0.05))),
        "earlystopping": EarlyStopping(monitor="val_loss", patience=3),
    }
    if RUNNING_ON_HPC:
        callbacks.pop("tqdm")

    wandb_logger = pl_loggers.WandbLogger(
        name=kwargs["name"],
        project=kwargs["project"],
        group=kwargs["group"] + ("-debug" if kwargs["debug"] else ""),
    )

    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=kwargs["max_epochs"] if not kwargs["debug"] else 11,
        callbacks=list(callbacks.values()),
        logger=[csv_logger, wandb_logger],
        check_val_every_n_epoch=kwargs["val_freq"],
        fast_dev_run=kwargs["fast_dev_run"],
    )

    logger.info("Start training")
    trainer.fit(mlp, train_dataloader, valid_dataloader)
    logger.info("Training completed.")
    logger.info(f"Log: {log_dir}")


if __name__ == "__main__":
    train()
