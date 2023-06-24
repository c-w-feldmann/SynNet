"""Common methods and params shared by all models.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import yaml
from scipy import sparse

from synnet.encoding.distances import cosine_distance
from synnet.models.mlp import MLP
from synnet.MolEmbedder import MolEmbedder

logger = logging.getLogger(__file__)


def init_save_dir(root_log_dir: str, suffix: Optional[str] = None) -> Path:
    """Creates folder with timestamp: `$path/<timestamp>$suffix`."""
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{suffix}" if suffix else ""
    log_dir = Path(root_log_dir) / (now + suffix)

    log_dir.mkdir(exist_ok=True, parents=True)
    logger.debug(f"Created folder `{log_dir}`")
    return log_dir


def _compute_class_weights_from_dataloader(dataloader, as_tensor: bool = False):
    from sklearn.utils.class_weight import compute_class_weight

    y: torch.Tensor = dataloader.dataset.tensors[-1]
    classes = y.unique().numpy()
    y = y.numpy()
    class_weight = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    if as_tensor:
        class_weight = torch.from_numpy(class_weight)
    return class_weight


def _fetch_molembedder(file: str):
    logger.info(f"Try to load precomputed MolEmbedder from {file}.")
    molembedder = MolEmbedder().load_precomputed(file).init_balltree(metric=cosine_distance)
    logger.info(f"Loaded MolEmbedder from {file}.")
    return molembedder


def load_mlp_from_ckpt(ckpt_file: str):
    """Load a model from a checkpoint for inference."""
    try:
        model = MLP.load_from_checkpoint(ckpt_file)
    except TypeError:
        model = _load_mlp_from_iclr_ckpt(ckpt_file)
    return model.eval()


def find_best_model_ckpt(path: str) -> Union[Path, None]:
    """Find checkpoint with lowest val_loss.

    Poor man's regex:
    somepath/act/ckpts.epoch=70-val_loss=0.03.ckpt
                                         ^^^^--extract this as float
    """
    ckpts = Path(path).rglob("*.ckpt")
    best_model_ckpt = None
    lowest_loss = 10_000  # ~ math.inf
    for file in ckpts:
        stem = file.stem
        val_loss = float(stem.split("val_loss=")[-1])
        if val_loss < lowest_loss:
            best_model_ckpt = file
            lowest_loss = val_loss
    return best_model_ckpt


def _load_mlp_from_iclr_ckpt(ckpt_file: str):
    """Load a model from a checkpoint for inference.
    Info: hparams were not saved, so we specify the ones needed for inference again."""
    model = Path(ckpt_file).parent.name  # assume "<dirs>/<model>/<file>.ckpt"
    kwargs = {
        "num_dropout_layers": 1,
        "optimizer": "adam",
        "learning_rate": 1e-4,
        "val_freq": 10,
    }
    if model == "act":
        model = MLP.load_from_checkpoint(
            ckpt_file,
            input_dim=3 * 4096,
            output_dim=4,
            hidden_dim=1000,
            num_layers=5,
            task="classification",
            dropout=0.5,
            loss="cross_entropy",
            valid_loss="accuracy",
            **kwargs,
        )
    elif model == "rt1":
        model = MLP.load_from_checkpoint(
            ckpt_file,
            input_dim=3 * 4096,
            output_dim=256,
            hidden_dim=1200,
            num_layers=5,
            task="regression",
            dropout=0.5,
            loss="mse",
            valid_loss="mse",  # Info: Used to be accuracy on kNN in embedding space, but that's very slow
            **kwargs,
        )
    elif model == "rxn":
        model = MLP.load_from_checkpoint(
            ckpt_file,
            input_dim=4 * 4096,
            output_dim=91,
            hidden_dim=3000,
            num_layers=5,
            task="classification",
            dropout=0.5,
            loss="mse",
            valid_loss="mse",  # Info: Used to be accuracy on kNN in embedding space, but that's very slow
            **kwargs,
        )
    elif model == "rt2":
        model = MLP.load_from_checkpoint(
            ckpt_file,
            input_dim=4 * 4096 + 91,
            output_dim=256,
            hidden_dim=3000,
            num_layers=5,
            task="regression",
            dropout=0.5,
            loss="mse",
            valid_loss="mse",  # Info: Used to be accuracy on kNN in embedding space, but that's very slow
            **kwargs,
        )

    else:
        raise ValueError
    return model.eval()


def asdict(obj) -> dict:
    return {k: v for k, v in obj.__dict__.items() if not k.startswith("__")}


def _download_to_file(url: str, filename: str) -> None:
    import requests
    from tqdm import tqdm

    # Adapted  from: https://stackoverflow.com/a/62113293 & https://stackoverflow.com/a/16696317
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))

        with tqdm(total=total_size, desc=Path(filename).name, unit="iB", unit_scale=True) as pbar:
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 4):
                    pbar.update(len(chunk))
                    f.write(chunk)


def download_iclr_checkpoint():
    """Download checkpoints as described in ICLR submission."""
    import shutil
    import tarfile

    from tqdm import tqdm

    CHECKPOINT_URL = "https://figshare.com/ndownloader/files/31067692"
    CHECKPOINT_FILE = "hb_fp_2_4096_256.tar.gz"
    CHECKPOINTS_DIR = "checkpoints/iclr/"

    CHECKPOINTS_DIR = Path(CHECKPOINTS_DIR)
    # Download
    if not Path(CHECKPOINT_FILE).exists():
        _download_to_file(CHECKPOINT_URL, CHECKPOINT_FILE)

    # Extract downloaded file
    CHECKPOINTS_DIR.mkdir(exist_ok=True, parents=True)
    with tarfile.open(CHECKPOINT_FILE) as tar:
        for member in tqdm(tar.getmembers(), desc=f"Extracting {CHECKPOINT_FILE}"):
            tar.extract(member, path=CHECKPOINTS_DIR)

    # Rename files to match new scripts
    for file in CHECKPOINTS_DIR.rglob("hb_fp_2_4096_256/*.ckpt"):
        model = file.stem

        new_file = CHECKPOINTS_DIR / f"{model}/{model}-dummy-val_loss=0.03.ckpt"
        new_file.parent.mkdir(exist_ok=True, parents=True)

        shutil.copy(file, new_file)

    # Clenup
    shutil.rmtree(CHECKPOINTS_DIR / "hb_fp_2_4096_256")

    print(f"Successfully downloaded files to {CHECKPOINTS_DIR}")


if __name__ == "__main__":
    pass
