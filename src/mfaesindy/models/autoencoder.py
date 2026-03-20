"""Multi-fidelity autoencoder building blocks."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def _mlp(dims: Iterable[int]) -> nn.Sequential:
    sizes = list(dims)
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(sizes[:-2], sizes[1:-1]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.GELU())
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int) -> None:
        super().__init__()
        self.network = _mlp([input_dim, *hidden_dims, latent_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: list[int], output_dim: int) -> None:
        super().__init__()
        self.network = _mlp([latent_dim, *hidden_dims, output_dim])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class MultiFidelityAutoencoder(nn.Module):
    """Separate LF/HF pathways with a shared latent dimensionality.

    The current implementation keeps the public interface stable while the
    internals remain intentionally simple. Partial weight-sharing can be added
    later without changing downstream training code.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.encoder_lf = Encoder(input_dim, hidden_dims, latent_dim)
        self.encoder_hf = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder_lf = Decoder(latent_dim, list(reversed(hidden_dims)), input_dim)
        self.decoder_hf = Decoder(latent_dim, list(reversed(hidden_dims)), input_dim)

    def encode_lf(self, x_lf: torch.Tensor) -> torch.Tensor:
        return self.encoder_lf(x_lf)

    def encode_hf(self, x_hf: torch.Tensor) -> torch.Tensor:
        return self.encoder_hf(x_hf)

    def decode_lf(self, z_lf: torch.Tensor) -> torch.Tensor:
        return self.decoder_lf(z_lf)

    def decode_hf(self, z_hf: torch.Tensor) -> torch.Tensor:
        return self.decoder_hf(z_hf)

    def forward(self, x_lf: torch.Tensor, x_hf: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z_lf = self.encode_lf(x_lf)
        outputs: dict[str, torch.Tensor] = {
            "z_lf": z_lf,
            "xhat_lf": self.decode_lf(z_lf),
        }
        if x_hf is not None:
            z_hf = self.encode_hf(x_hf)
            outputs["z_hf"] = z_hf
            outputs["xhat_hf"] = self.decode_hf(z_hf)
        return outputs
