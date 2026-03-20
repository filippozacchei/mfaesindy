"""Training scaffold for multi-fidelity latent dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mfaesindy.config import ExperimentConfig
from mfaesindy.data.types import TrajectoryBatch
from mfaesindy.losses import MultiFidelityLoss
from mfaesindy.models import MultiFidelityAutoencoder


@dataclass(slots=True)
class TrainingState:
    epoch: int = 0
    best_loss: float = float("inf")


class Trainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.model = MultiFidelityAutoencoder(
            input_dim=config.autoencoder.input_dim,
            latent_dim=config.autoencoder.latent_dim,
            hidden_dims=config.autoencoder.hidden_dims,
        )
        self.loss_fn = MultiFidelityLoss(
            lambda_align=config.loss.lambda_align,
            lambda_dyn=config.loss.lambda_dyn,
            lambda_mf=config.loss.lambda_mf,
            lambda_smooth=config.loss.lambda_smooth,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
        )
        self.state = TrainingState()

    def train_step(self, batch: TrajectoryBatch) -> dict[str, float]:
        self.model.train()
        outputs = self.model(batch.x_lf, batch.x_hf)

        loss_rec = self.loss_fn.reconstruction(
            x_lf=batch.x_lf,
            xhat_lf=outputs["xhat_lf"],
            x_hf=batch.x_hf,
            xhat_hf=outputs.get("xhat_hf"),
        )
        loss_align = self.loss_fn.alignment(outputs["z_lf"], outputs.get("z_hf"))
        loss = loss_rec + self.loss_fn.lambda_align * loss_align

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "reconstruction": float(loss_rec.detach().cpu()),
            "alignment": float(loss_align.detach().cpu()),
        }
