"""
Multi-layer perceptron (MLP) class.
"""
import logging
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import sklearn.neighbors as skl_nn
import torch
import torch.nn.functional as torch_func
from torch import nn

from synnet.encoding.embedding import MolecularEmbeddingManager

logger = logging.getLogger(__name__)


class MLP(pl.LightningModule):
    TRAIN_LOSSES = "cross_entropy mse l1 huber cosine_distance".split()
    VALID_LOSSES = TRAIN_LOSSES + "accuracy nn_accuracy".split()
    OPTIMIZERS = "sgd adam".lower().split()
    device: torch.device

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_dropout_layers: int,
        task: str,
        loss: str,
        valid_loss: str,
        optimizer: str,
        learning_rate: float,
        val_freq: int,
        ncpu: Optional[int] = None,
        molembedder: Optional[MolecularEmbeddingManager] = None,  # for knn-accuracy
        class_weights: Optional[npt.NDArray[np.float_]] = None,
    ) -> None:
        if loss not in self.TRAIN_LOSSES:
            raise ValueError(f"Unsupported loss function {loss}")
        if valid_loss not in self.VALID_LOSSES:
            raise ValueError(f"Unsupported loss function {valid_loss}")
        if optimizer not in self.OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer {optimizer}")
        if num_dropout_layers > num_layers - 2:
            raise Warning("Requested more dropout layers than there are hidden layers.")
        if class_weights is not None and task == "regression":
            raise Warning(f"Provided argument `{class_weights=}` for a regression task")

        super().__init__()
        self.save_hyperparameters(ignore="molembedder")

        self.loss = loss
        self.valid_loss = valid_loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.ncpu = ncpu  # unused
        self.val_freq = val_freq
        self.molembedder = molembedder
        self.class_weights = class_weights
        self.task = task

        # Create modules
        modules: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ]

        # Input layer

        # Hidden layers
        for i in range(num_layers - 2):  # "-2" for first & last layer
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.BatchNorm1d(hidden_dim))
            modules.append(nn.ReLU())
            # Add dropout, starting from last layer
            if i >= num_layers - 2 - num_dropout_layers:
                modules.append(nn.Dropout(dropout))

        # Output layer
        modules.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward step for inference only."""
        y_hat = self.layers(x)

        # During training, `cross_entropy` loss expects raw logits.
        # We add the softmax here so that mlp.forward(X) can be used for inference.
        if self.hparams.task == "classification":
            y_hat = torch_func.softmax(y_hat, dim=-1)
        return y_hat

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """The complete training loop."""
        x, y = batch
        y_hat = self.layers(x)
        if self.loss == "cross_entropy":
            weights = (
                torch.tensor(self.class_weights, device=self.device, dtype=y_hat.dtype)
                if self.class_weights is not None
                else None
            )
            loss = torch_func.cross_entropy(y_hat, y.long(), weight=weights)
        elif self.loss == "mse":
            loss = torch_func.mse_loss(y_hat, y)
        elif self.loss == "l1":
            loss = torch_func.l1_loss(y_hat, y)
        elif self.loss == "huber":
            loss = torch_func.huber_loss(y_hat, y)
        elif self.loss == "cosine_distance":
            loss = 1 - torch_func.cosine_similarity(y, y_hat).mean()
        else:
            raise NotImplementedError(f"Loss function '{self.loss}' is not available.")

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """The complete validation loop."""
        x, y = batch
        y_hat = self.layers(x)
        if self.valid_loss == "cross_entropy":
            weights = (
                torch.tensor(self.class_weights, device=self.device, dtype=y_hat.dtype)
                if self.class_weights is not None
                else None
            )
            loss = torch_func.cross_entropy(y_hat, y.long(), weight=weights)
        elif self.valid_loss == "accuracy":
            y_hat = torch.argmax(y_hat, dim=1)
            accuracy = (y_hat == y).sum() / len(y)
            loss = 1 - accuracy
        elif self.valid_loss == "nn_accuracy":
            # NOTE: Very slow!
            # Performing the knn-search can easily take a couple of minutes,
            # even for small datasets.
            if self.molembedder is None:
                raise ValueError("No `molembedder` provided for knn-accuracy.")
            kdtree = self.molembedder.kdtree
            y_nn = nn_search_list(y.detach().cpu().numpy(), kdtree)
            y_hat = nn_search_list(y_hat.detach().cpu().numpy(), kdtree)

            accuracy = (y_hat == y_nn).sum() / len(y_nn)
            loss = 1 - accuracy
        elif self.valid_loss == "mse":
            loss = torch_func.mse_loss(y_hat, y)
        elif self.valid_loss == "l1":
            loss = torch_func.l1_loss(y_hat, y)
        elif self.valid_loss == "huber":
            loss = torch_func.huber_loss(y_hat, y)
        elif self.valid_loss == "cosine_distance":
            loss = 1 - torch_func.cosine_similarity(y, y_hat).mean()
        else:
            raise NotImplementedError(f"Loss function '{self.valid_loss}' is not available.")

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Define Optimerzers and LR schedulers."""
        optimizer: torch.optim.Optimizer
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError(f"Optimizer '{self.optimizer}' is not available.")
        # if (lr_scheduler := self.hparams.get("lr_scheduler_config",None)) is not None:
        # out["lr_scheduler"] = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1,verbose=True)

        return {"optimizer": optimizer}


def nn_search_list(y: npt.NDArray[np.float_], kdtree: skl_nn.KDTree) -> npt.NDArray[np.int_]:
    y = np.atleast_2d(y)  # (n_samples, n_features)
    ind = kdtree.query(y, k=1, return_distance=False)  # (n_samples, 1)
    return ind


if __name__ == "__main__":
    pass
