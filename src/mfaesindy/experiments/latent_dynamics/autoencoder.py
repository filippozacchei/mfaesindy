"""Autoencoder models for studying encoder sharing across fidelities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _make_mlp(widths: list[int], final_activation: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (in_features, out_features) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(in_features, out_features))
        is_last = index == len(widths) - 2
        if not is_last or final_activation:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


@dataclass(slots=True)
class AutoencoderArchitecture:
    input_dim: int
    latent_dim: int = 2
    private_widths: tuple[int, ...] = (128, 64)
    shared_widths: tuple[int, ...] = ()
    decoder_widths: tuple[int, ...] = (64, 128)


class FidelityEncoder(nn.Module):
    def __init__(self, input_dim: int, private_widths: tuple[int, ...], shared_tail: nn.Module | None, latent_dim: int) -> None:
        super().__init__()
        self.private = _make_mlp([input_dim, *private_widths]) if private_widths else nn.Identity()
        tail_input = private_widths[-1] if private_widths else input_dim
        self.tail = shared_tail if shared_tail is not None else _make_mlp([tail_input, latent_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.private(x))


class FidelityDecoder(nn.Module):
    def __init__(self, latent_dim: int, shared_head: nn.Module | None, decoder_widths: tuple[int, ...], output_dim: int) -> None:
        super().__init__()
        if shared_head is None:
            head_output = decoder_widths[0] if decoder_widths else output_dim
            self.head = _make_mlp([latent_dim, head_output])
            tail_input = head_output
            tail_widths = decoder_widths[1:]
        else:
            self.head = shared_head
            tail_input = decoder_widths[0] if decoder_widths else latent_dim
            tail_widths = decoder_widths[1:] if decoder_widths else ()

        if decoder_widths:
            self.tail = _make_mlp([tail_input, *tail_widths, output_dim])
        else:
            self.tail = nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.tail(self.head(z))


class MultiFidelityAutoencoder(nn.Module):
    """Two-fidelity autoencoder with configurable encoder/decoder sharing."""

    def __init__(
        self,
        architecture: AutoencoderArchitecture,
        share_encoder_tail: bool = False,
        share_decoder_head: bool = False,
    ) -> None:
        super().__init__()
        tail_input = architecture.private_widths[-1] if architecture.private_widths else architecture.input_dim

        shared_encoder_tail = None
        if share_encoder_tail:
            shared_encoder_tail = _make_mlp([tail_input, *architecture.shared_widths, architecture.latent_dim])

        shared_decoder_head = None
        if share_decoder_head:
            head_output = architecture.decoder_widths[0] if architecture.decoder_widths else architecture.input_dim
            shared_decoder_head = _make_mlp([architecture.latent_dim, head_output])

        self.encoder_lf = FidelityEncoder(
            input_dim=architecture.input_dim,
            private_widths=architecture.private_widths,
            shared_tail=shared_encoder_tail,
            latent_dim=architecture.latent_dim,
        )
        self.encoder_hf = FidelityEncoder(
            input_dim=architecture.input_dim,
            private_widths=architecture.private_widths,
            shared_tail=shared_encoder_tail,
            latent_dim=architecture.latent_dim,
        )
        self.decoder_lf = FidelityDecoder(
            latent_dim=architecture.latent_dim,
            shared_head=shared_decoder_head,
            decoder_widths=architecture.decoder_widths,
            output_dim=architecture.input_dim,
        )
        self.decoder_hf = FidelityDecoder(
            latent_dim=architecture.latent_dim,
            shared_head=shared_decoder_head,
            decoder_widths=architecture.decoder_widths,
            output_dim=architecture.input_dim,
        )

    def forward(self, x_lf: torch.Tensor, x_hf: torch.Tensor) -> dict[str, torch.Tensor]:
        z_lf = self.encoder_lf(x_lf)
        z_hf = self.encoder_hf(x_hf)
        return {
            "z_lf": z_lf,
            "z_hf": z_hf,
            "xhat_lf": self.decoder_lf(z_lf),
            "xhat_hf": self.decoder_hf(z_hf),
        }
