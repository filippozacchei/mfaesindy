from __future__ import annotations

import torch
from torch import nn

from .base import BaseAutoencoder


class SharedLatentAutoencoder(BaseAutoencoder):
    def __init__(
        self,
        encoder_stems: dict[str, nn.Module],
        shared_encoder: nn.Module,
        to_latent: nn.Module,
        from_latent: nn.Module,
        shared_decoder: nn.Module,
        decoder_heads: dict[str, nn.Module],
        latent_dim: int,
    ) -> None:
        fidelities = tuple(encoder_stems.keys())
        super().__init__(fidelities=fidelities)

        if set(encoder_stems) != set(decoder_heads):
            raise ValueError(
                "encoder_stems and decoder_heads must have the same fidelities."
            )

        self.encoder_stems = nn.ModuleDict(encoder_stems)
        self.shared_encoder = shared_encoder
        self.to_latent = to_latent
        self.from_latent = from_latent
        self.shared_decoder = shared_decoder
        self.decoder_heads = nn.ModuleDict(decoder_heads)
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor, fidelity: str) -> torch.Tensor:
        self._check_fidelity(fidelity)
        h = self.encoder_stems[fidelity](x)
        h = self.shared_encoder(h)
        z = self.to_latent(h)
        return z

    def decode(self, z: torch.Tensor, fidelity: str) -> torch.Tensor:
        self._check_fidelity(fidelity)
        h = self.from_latent(z)
        h = self.shared_decoder(h)
        x_hat = self.decoder_heads[fidelity](h)
        return x_hat
