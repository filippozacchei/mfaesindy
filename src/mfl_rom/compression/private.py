from __future__ import annotations

import torch
from torch import nn

from .base import BaseAutoencoder


class SharedPrivateLatentAutoencoder(BaseAutoencoder):
    def __init__(
        self,
        encoder_stems: dict[str, nn.Module],
        shared_encoder: nn.Module,
        to_shared: nn.Module,
        to_private: dict[str, nn.Module],
        from_latent: nn.Module,
        shared_decoder: nn.Module,
        decoder_heads: dict[str, nn.Module],
        shared_dim: int,
        private_dim: int,
    ) -> None:
        fidelities = tuple(encoder_stems.keys())
        super().__init__(fidelities=fidelities)

        if set(encoder_stems) != set(decoder_heads):
            raise ValueError(
                "encoder_stems and decoder_heads must have the same fidelities."
            )
        if set(encoder_stems) != set(to_private):
            raise ValueError(
                "encoder_stems and to_private must have the same fidelities."
            )

        self.encoder_stems = nn.ModuleDict(encoder_stems)
        self.shared_encoder = shared_encoder
        self.to_shared = to_shared
        self.to_private = nn.ModuleDict(to_private)
        self.from_latent = from_latent
        self.shared_decoder = shared_decoder
        self.decoder_heads = nn.ModuleDict(decoder_heads)
        self.shared_dim = shared_dim
        self.private_dim = private_dim

    def split_latent(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_shared = z[..., : self.shared_dim]
        z_private = z[..., self.shared_dim :]
        return z_shared, z_private

    def encode(self, x: torch.Tensor, fidelity: str) -> torch.Tensor:
        self._check_fidelity(fidelity)
        h = self.encoder_stems[fidelity](x)
        h = self.shared_encoder(h)
        z_shared = self.to_shared(h)
        z_private = self.to_private[fidelity](h)
        return torch.cat([z_shared, z_private], dim=-1)

    def decode(self, z: torch.Tensor, fidelity: str) -> torch.Tensor:
        self._check_fidelity(fidelity)
        h = self.from_latent(z)
        h = self.shared_decoder(h)
        x_hat = self.decoder_heads[fidelity](h)
        return x_hat
