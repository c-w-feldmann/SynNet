"""Common methods and params shared by all models."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.utils.data as torch_data
import yaml
from loguru import logger
from scipy import sparse

from synnet.encoding.embedding import MolecularEmbeddingManager
from synnet.models.mlp import MLP
from synnet.utils.custom_types import PathType


def init_save_dir(path: PathType, suffix: str = "") -> Path:
    """Creates folder with timestamp: `$path/<timestamp>$suffix`."""
    now = datetime.now().strftime("%Y_%m_%d-%H%M%S")
    save_dir = Path(path) / (now + suffix)

    save_dir.mkdir(exist_ok=True, parents=True)
    return save_dir


def load_config_file(file: PathType) -> dict[str, Union[str, int]]:
    """Load a `*.yaml`-config file."""
    file = Path(file)
    if not file.suffix == ".yaml":
        raise NotImplementedError(f"Can only read config from yaml file, not {file}.")
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config


def xy_to_dataloader(
    X_file: PathType,
    y_file: PathType,
    task: str,
    n: Optional[Union[int, float]] = 1.0,
    **kwargs: Any,
) -> torch_data.DataLoader:  # type: ignore[type-arg]
    """Loads featurized X,y `*.npz`-data into a `DataLoader`"""
    X = sparse.load_npz(X_file)
    y = sparse.load_npz(y_file)
    # Filer?
    if isinstance(n, int):
        if n > X.shape[0]:
            logger.warning(
                f"n={n} exceeds size of dataset {X.shape[0]}. "
                f"Setting n to {X.shape[0]}."
            )
            n = int(X.shape[0])
        X = X[:n]
        y = y[:n]
    elif isinstance(n, float) and n < 1.0:
        xn = X.shape[0] * n
        yn = X.shape[0] * n
        X = X[:xn]
        y = y[:yn]
    else:
        pass  #
    X = np.atleast_2d(np.asarray(X.todense()))
    y = (
        np.atleast_2d(np.asarray(y.todense()))
        if task == "regression"
        else np.asarray(y.todense()).squeeze()
    )
    dataset = torch.utils.data.TensorDataset(
        torch.Tensor(X),
        torch.Tensor(y),
    )
    logger.info(f"Loaded {X_file}, {X.shape=}")
    logger.info(f"Loaded {y_file}, {y.shape=}")
    return torch_data.DataLoader(dataset, **kwargs)  # type: ignore[arg-type]


def _compute_class_weights_from_dataloader(
    dataloader: torch_data.DataLoader, as_tensor: bool = False  # type: ignore[type-arg]
) -> npt.NDArray[np.float_]:
    from sklearn.utils.class_weight import compute_class_weight

    if not hasattr(dataloader.dataset, "tensors"):
        raise AssertionError(
            "Dataloader must have a dataset with a `tensors`-attribute."
        )
    y: torch.Tensor = dataloader.dataset.tensors[-1]
    classes = y.unique().numpy()
    y = y.numpy()
    class_weight = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    if as_tensor:
        class_weight = torch.from_numpy(class_weight)
    return class_weight


def _fetch_molembedder(folder: PathType) -> MolecularEmbeddingManager:
    logger.info(f"Try to load precomputed MolEmbedder from {folder}.")
    molembedder = MolecularEmbeddingManager.from_folder(folder)
    logger.info(f"Loaded MolEmbedder from {folder}.")
    return molembedder


def load_mlp_from_ckpt(ckpt_file: PathType) -> MLP:
    """Load a model from a checkpoint for inference."""
    try:
        model = MLP.load_from_checkpoint(ckpt_file)
    except TypeError:
        model = _load_mlp_from_iclr_ckpt(ckpt_file)
    return model.eval()


def find_best_model_ckpt(path: PathType) -> Path:
    """Find checkpoint with lowest val_loss.

    Poor man's regex:
    somepath/act/ckpts.epoch=70-val_loss=0.03.ckpt
                                         ^^^^--extract this as float
    """
    ckpts = Path(path).rglob("*.ckpt")
    best_model_ckpt = None
    lowest_loss = 10000.0  # ~ math.inf
    for file in ckpts:
        stem = file.stem
        val_loss = float(stem.split("val_loss=")[-1])
        if val_loss < lowest_loss:
            best_model_ckpt = file
            lowest_loss = val_loss
    if best_model_ckpt is None:
        raise FileNotFoundError(f"Could not find any checkpoint in {path}.")
    return best_model_ckpt


def _load_mlp_from_iclr_ckpt(ckpt_file: PathType) -> MLP:
    """Load a model from a checkpoint for inference.
    Info: hparams were not saved, so we specify the ones needed for inference again."""
    model_name = Path(ckpt_file).parent.name  # assume "<dirs>/<model>/<file>.ckpt"
    kwargs: dict[str, Any] = {
        "num_dropout_layers": 1,
        "optimizer": "adam",
        "learning_rate": 1e-4,
        "val_freq": 10,
    }
    if model_name == "act":
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
    elif model_name == "rt1":
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
    elif model_name == "rxn":
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
    elif model_name == "rt2":
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


def asdict(obj: Any) -> dict[str, Any]:
    return {k: v for k, v in obj.__dict__.items() if not k.startswith("__")}


def _download_to_file(url: str, filename: str) -> None:
    import requests
    from tqdm import tqdm

    # Adapted  from: https://stackoverflow.com/a/62113293 & https://stackoverflow.com/a/16696317
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))

        with tqdm(
            total=total_size, desc=Path(filename).name, unit="iB", unit_scale=True
        ) as pbar:
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 4):
                    pbar.update(len(chunk))
                    f.write(chunk)


def download_iclr_checkpoint() -> None:
    """Download checkpoints as described in ICLR submission."""
    import shutil
    import tarfile

    from tqdm import tqdm

    CHECKPOINT_URL = "https://figshare.com/ndownloader/files/31067692"
    CHECKPOINT_FILE = "hb_fp_2_4096_256.tar.gz"
    CHECKPOINTS_DIR = Path("checkpoints/iclr/")
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
