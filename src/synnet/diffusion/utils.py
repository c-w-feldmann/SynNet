import logging
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class BetaSchedulesConfig:
    def __init__(
        self,
        beta_schedule: str = "linear",
        start_beta: float = 0.0001,
        end_beta: float = 0.02,
        cosine_s: float = 0.008,
        **kwargs: dict,
    ):
        self.beta_schedule = beta_schedule
        self.start_beta = start_beta
        self.end_beta = end_beta
        if beta_schedule == "cosine":
            self.cosine_s = cosine_s
        self.kwargs = kwargs

    @property
    def hparams(self):
        return asdict(self)


class BetaSchedules:
    def __init__(self, schedule: Optional[str] = "linear") -> None:
        schedule_dict: dict[str, Callable] = {
            "cosine": self.cosine_beta_schedule,
            "linear": self.linear_beta_schedule,
            "quadratic": self.quadratic_beta_schedule,
            "sigmoid": self.sigmoid_beta_schedule,
        }
        self._schedules = list(schedule_dict.keys())
        self.schedule = schedule_dict[schedule]

    @property
    def schedules(self):
        return self._schedules

    def __call__(self, timesteps: int) -> torch.tensor:
        return self.schedule(timesteps)

    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    @staticmethod
    def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    @staticmethod
    def quadratic_beta_schedule(timesteps, start=0.0001, end=0.02):
        return torch.linspace(start**0.5, end**0.5, timesteps) ** 2

    @staticmethod
    def sigmoid_beta_schedule(timesteps, start=0.0001, end=0.02):
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (end - start) + start


def extract(array, t, x_shape):
    """Extracts coefficients at specified timesteps,
    then reshape for broadcasting purposes.
    """
    device = array.device
    batch_size = t.shape[0]
    t = t.to(device)
    out = array.gather(-1, t)
    # assert len(x_shape) > 1, "x must be at least 2D, did you forget to add batch dimension?"
    logger.debug(f"`extract()`: Moving to {device}")
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(device)
