import logging
import math
from typing import Callable, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

import wandb
from synnet.diffusion.utils import BetaSchedules, extract

logger = logging.getLogger(__name__)


class Linear(nn.Module):
    """Linear layer with activation"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation=nn.Softplus(),
        batch_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._has_bn = batch_norm
        self._has_dropout = dropout > 0.0

        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        if self._has_bn:
            self.bn = nn.BatchNorm1d(out_features)
        if self._has_dropout:
            self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass"""
        # print(f"Forward pass on x: {x.shape=}")
        output = self.linear(x)
        if self._has_bn:
            output = self.bn(output)
        output = self.activation(output)
        if self._has_dropout:
            output = self.dropout_layer(output)
        return output


class MLP(nn.Module):
    """Simple MLP model"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: Union[int, tuple[int]],
        num_hidden_layers: Optional[int] = None,
        dropout: float = 0.0,
        batch_norm: bool = True,
        loss: str = "MSE",
        val_loss: str = "MSE",
        act_fct: Callable = nn.Softplus(),
        learning_rate: float = 3e-4,
        positional_encoder_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        # self.save_hyperparameters(ignore=["positional_encoder"]) # INFO: only for pl        self.automatic_optimization = False

        if positional_encoder_kwargs is not None:
            self.positional_encoder = PositionalEncoder(**positional_encoder_kwargs)
        else:
            self.positional_encoder = None  # TODO: find more elegant way
        # Hidden layers
        if isinstance(hidden_size, int):
            if num_hidden_layers is None:
                raise ValueError(f"Must specify number of hidden layers when `hidden_size` is int")
            _hidden_sizes: tuple[int] = (hidden_size,) * num_hidden_layers
        else:
            _hidden_sizes = hidden_size

        # Loss functions
        self.get_loss_fct(loss), self.get_loss_fct(val_loss)  # input check
        self._loss = loss
        self._val_loss = val_loss

        # Layers
        layers = list()
        #   Input layer
        layers += [Linear(input_size, _hidden_sizes[0])]
        #   Hidden layers
        for _in, _out in zip(_hidden_sizes[:-1], _hidden_sizes[1:]):
            layers += [
                Linear(
                    _in,
                    _out,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    activation=act_fct,
                )
            ]
        #   Output layer
        # self.output_layer = Linear(hidden_size, output_size, activation=nn.Identity())
        layers += [Linear(_hidden_sizes[-1], output_size, activation=nn.Identity())]
        # All layers together
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        # Sanity checks
        # if t is not None and self.positional_encoder is None:
        #     raise ValueError(f"No positional encoder provided, but provided timesteps ({t=}).")
        if t is None and self.positional_encoder is not None:
            raise ValueError(
                f"Positional encoder {self.positional_encoder} provided, but no timesteps provided."
            )
        logger.debug(f"Forward pass on x: {x.shape=}")
        if self.positional_encoder is not None:
            t_emb = self.positional_encoder.forward(t)
            logger.debug(f"Forward pass on t_emb: {t_emb.shape=}")
            mode = self.positional_encoder.kwargs["mode"]
            denoise = self.positional_encoder.kwargs["denoise"]
            logger.debug(f"PE {mode=}, {denoise=}")
            if denoise == "xy":
                if mode == "add":
                    x = x + t_emb
                elif mode == "mul":
                    x = x * t_emb
                elif mode == "cat":
                    # INFO: embarrassing design, we always pass in (x,y) but we had set the input dim to be (x,y) + emb_t
                    x = torch.cat((x, t_emb), axis=1)
            elif denoise == "y":
                bs = t_emb.shape[0]
                dim_x = self.positional_encoder.kwargs["dim_x"]
                if mode == "add":
                    t_emb = torch.concat((torch.zeros((bs, dim_x)), t_emb), axis=1)
                    x = x + t_emb
                if mode == "mul":
                    t_emb = torch.concat((torch.ones((bs, dim_x)), t_emb), axis=1)
                    x = x * t_emb
                if mode == "cat":
                    x = torch.cat((x, t_emb), axis=1)
        return self.layers(x)

    # region LightningModule
    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     loss = self.loss_fct(y_hat, y)
    #     self.log("train_loss", loss)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     loss = self.val_loss_fct(y_hat, y)
    #     self.log("val_loss", loss)
    #     return loss

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     loss = self.val_loss_fct(y_hat, y)
    #     self.log("test_loss", loss)
    #     return loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams_initial["learning_rate"])

    #     decay_rate = 0.8
    #     start_lr = self.learning_rate
    #     min_lr = 1.0e-05
    #     steps_to_decay = 1
    #     min_decay_rate = min_lr / start_lr
    #     lr_lambda = lambda epoch: (
    #         np.maximum(decay_rate ** (epoch // steps_to_decay), min_decay_rate)
    #     )
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    #     return {"optimizer": optimizer, "lr_scheduler": scheduler}
    # endregion LightningModule

    def get_loss_fct(self, loss: str):
        loss = loss.lower()
        if loss == "mse":
            return nn.MSELoss()
        elif loss == "mae" or loss == "l1":
            return nn.L1Loss()
        elif loss == "cross_entropy":
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function {loss}")

    @property
    def loss_fct(self):
        return self.get_loss_fct(self._loss)

    @property
    def val_loss_fct(self):
        return self.get_loss_fct(self._val_loss)


class PositionalEncoder(nn.Module):
    """Positional encoding as described in "Attention is all you need",
    copied from
     - https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    """

    def __init__(
        self,
        d_model: int,  # dimension of output
        dropout: float = 0.0,
        max_timesteps: int = 1000,  # "max_len"
        n=10000.0,  # arbitrary factor, same as in AIAYN
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs | {
            "d_model": d_model,
            "dropout": dropout,
            "max_timesteps": max_timesteps,
            "n": n,
        }

        self._with_dropout = dropout > 0.0
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_timesteps).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(n) / d_model))

        pe = torch.zeros(max_timesteps, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        #                                           ^ fix for d_model odd
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor, shape [batch_size]
            x: Optional: Tensor, shape [batch_size, embedding_dim]
        """
        logger.debug(f".forward(): {t.shape=}")
        t_emb = self.pe[t, :] * 0.2
        return self.dropout(t_emb) if self._with_dropout else t_emb

    def forward_with_x(self, x: torch.Tensor, t: torch.Tensor):
        t_emb = self.forward(t)
        return x + t_emb


class GaussionDiffusion(pl.LightningModule):
    def __init__(
        self,
        *,
        model: Optional[nn.Module] = None,  # diffusion model eps_theta(x,t)
        beta_schedule: str = "linear",
        denoise: str,
        loss: str = "mse",
        timesteps: int = 1000,
        learning_rate: float = 1e-3,
        **kwargs: dict,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        # Model to predict noise
        self.model = MLP(12288 + 256, 256, 3, num_hidden_layers=1)

        # Get beta schedule
        betas = BetaSchedules(schedule=beta_schedule)(timesteps=timesteps)
        self.init_constants(betas)
        self.num_timesteps = int(betas.shape[0])

        logger.debug(f"Initalized diffusion model with {self.num_timesteps} timesteps")

    def init_constants(self, betas: torch.Tensor):
        """Init all variation of alpha, alpha_bar, etc"""

        # define alphas
        alphas = 1.0 - betas  # alpha
        alphas_cumprod = torch.cumprod(alphas, axis=0)  #
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # Save to self
        self.betas = nn.Parameter(betas, requires_grad=False)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.alphas_cumprod = nn.Parameter(alphas_cumprod, requires_grad=False)
        self.alphas_cumprod_prev = nn.Parameter(alphas_cumprod_prev, requires_grad=False)
        self.sqrt_recip_alphas = nn.Parameter(sqrt_recip_alphas, requires_grad=False)
        self.sqrt_alphas_cumprod = nn.Parameter(sqrt_alphas_cumprod, requires_grad=False)
        self.sqrt_one_minus_alphas_cumprod = nn.Parameter(
            sqrt_one_minus_alphas_cumprod, requires_grad=False
        )
        self.posterior_variance = nn.Parameter(posterior_variance, requires_grad=False)

    @property
    def loss_fct(self):
        loss = self.hparams_initial["loss"]
        if loss == "l1":
            loss_fct = F.l1_loss
        elif loss == "l2" or loss == "mse":
            loss_fct = F.mse_loss
        elif loss == "huber":
            loss_fct = F.smooth_l1_loss
        else:
            raise NotImplementedError(f"Loss type {loss} not implemented.")
        return loss_fct

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Union[None, torch.Tensor] = None):
        """Forward diffusion pass to `t`.
        Compute q(xt | x0) = x_t(x0, eps)

        (Eq. 4 reparametrized, see just above Eq. 9)
        """
        if noise is None:
            noise = torch.randn_like(x0)  # randn_like samples from N(0,1)
        mean = extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
        std = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        return mean + std

    def p_lossses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        denoise = self.hparams["denoise"]
        if noise is None:
            if denoise == "xy":
                x_start = torch.cat([kwargs["x"], kwargs["y"]], dim=1)
            elif denoise == "y":
                x_start = kwargs["y"]
            logger.debug(f"Generating noise for {denoise} with shape `{x_start.shape}`")
            noise = torch.randn_like(x_start)

        if t is None:
            # t ~ Uniform({1,...,T})
            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            batch_size = x_start.shape[0]
            t = torch.randint(0, self.num_timesteps, (batch_size,)).long()
            logger.debug(f"Sampled timesteps, {t.shape=}, {t[:5]=}")

        # Sample x_t ~ q(x_t | x_0) = x_t(x_0, eps)
        #   -> applies noise to data (t steps forward)
        x_noisy = self.q_sample(x0=x_start, t=t, noise=noise)

        # Compute epsilon_theta(x_t, t)
        #   -> predict noise from data
        # INFO: Let us always learn the noise from a (noisy) joint distribution (x,y)
        #       model: (x,y) |-> noise
        if denoise == "xy":
            xy_noisy = x_noisy
        elif denoise == "y":
            xy_noisy = torch.cat([kwargs["x"], x_noisy], dim=1)

        predicted_noise = self.model(xy_noisy, t)

        # Compute loss
        loss = self.loss_fct(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x, t, **kwargs):
        """Reverse diffusion step.
        Sample x_{t-1} ~ p(x_{t-1} | x_t) (Alg. 2)"""
        # Extract consts
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        # Compute mean of model (Eq. 11)
        x_t = torch.concat([kwargs["x_data"], x], dim=1)  # TODO: if denoising over y, x=y here
        predicted_noise = self.model(x_t, t)

        mean_theta = sqrt_recip_alphas_t * (
            x - (betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        )
        if t[0] > 0:  # t is a tensor, but all elements are the same for inference
            # Sample noise
            noise = torch.randn_like(x)
            # Compute std
            std = torch.sqrt(extract(self.betas, t, x.shape)) * noise
            return mean_theta + std
        else:
            return mean_theta

    @torch.no_grad()
    def p_sample_loop(
        self,
        *,
        batch_size: int,
        dim_x: int,
        dim_y: int,
        x: Optional[torch.Tensor] = None,
        verbose: Optional[bool] = True,
        **kwargs,
    ):
        """Reverse diffusion process."""

        # assert (batch_size, dim_x) == x.shape

        denoise = self.hparams["denoise"]
        if denoise == "xy":
            raise NotImplementedError("TODO:")
            logger.debug(f"Begin from pure noise for x,y for {batch_size} samples.")
        elif denoise == "y":
            # Fix x, begin from pure noise for y
            logger.debug(
                f"Fixing x (d={dim_x}), begin from pure noise for y (d={dim_y}) for {batch_size} samples."
            )
            xgen = torch.randn((batch_size, dim_y))
        xgens = []

        loop = reversed(range(0, self.num_timesteps))
        if verbose:
            loop = tqdm(loop, desc="Sampling loop time step", total=self.num_timesteps)
        for t in loop:
            ts = torch.ones((batch_size,)).long() * t
            xgen = self.p_sample(xgen, ts, **kwargs)
            xgens.append(xgen.cpu().numpy())
        return np.asarray(xgens)

    def sample(self, x, dim_y: int, **kwargs):
        """Reverse diffusion process."""
        batch_size, dim_x = x.shape
        logger.debug(f"sample() {x.shape=}")

        return self.p_sample_loop(x=x, dim_x=dim_x, dim_y=dim_y, batch_size=batch_size, **kwargs)

    def forward(self, x, t):
        raise NotImplementedError()  # TODO: rethink design choices

    def training_step(self, batch, batch_idx):
        logger.debug("Training step")
        x, y = batch
        loss = self.p_lossses(None, **{"x": x, "y": y})
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        logger.debug("Validation step")
        x, y = batch
        loss = self.p_lossses(None, **{"x": x, "y": y})
        self.log("val_loss", loss)  # Note: do not change this name, lr scheduler monitors it
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.p_lossses(None, **{"x": x, "y": y})
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams_initial["learning_rate"])

        decay_rate = 0.9
        start_lr = self.hparams_initial["learning_rate"]
        min_lr = 1.0e-05
        steps_to_decay = 30
        min_decay_rate = min_lr / start_lr
        lr_lambda = lambda epoch: (
            np.maximum(decay_rate ** (epoch // steps_to_decay), min_decay_rate)
        )  #  A function which computes a multiplicative factor
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def plot_beta_schedules():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    TIMESTEPS = 1_000
    scheduler = BetaSchedules(schedule="linear")
    betas = scheduler(timesteps=TIMESTEPS)
    for schedule in BetaSchedules().schedules:
        scheduler = BetaSchedules(schedule=schedule)
        betas = scheduler(timesteps=TIMESTEPS)
        ax.plot(range(TIMESTEPS), betas, label=schedule)
    ax.legend()
    ax.set(xlabel="timestep", ylabel="value", yscale="log")
    return ax
