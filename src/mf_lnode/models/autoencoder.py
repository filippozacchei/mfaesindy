"""Shared and fidelity-specific encoder/decoder building blocks."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


def _activation_from_name(name: str) -> nn.Module:
    lookup: dict[str, Callable[[], nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return lookup[name.lower()]()
    except KeyError as error:
        raise ValueError(f"Unsupported activation '{name}'.") from error


def _build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_hidden_layers: int,
    activation: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_features = input_dim
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(_activation_from_name(activation))
        in_features = hidden_dim
    layers.append(nn.Linear(in_features, output_dim))
    return nn.Sequential(*layers)


class SharedEncoder(nn.Module):
    """Shared temporal encoder backbone operating on fidelity-adapted windows."""

    def __init__(
        self,
        window_size: int,
        adapter_dim: int,
        parameter_dim: int,
        hidden_dim: int,
        num_hidden_layers: int = 2,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.network = _build_mlp(
            input_dim=window_size * adapter_dim + parameter_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
        )

    def forward(self, adapted_sequence: Tensor, parameters: Tensor) -> Tensor:
        if adapted_sequence.ndim != 3:
            raise ValueError("adapted_sequence must have shape [batch, time, features].")
        if adapted_sequence.shape[1] != self.window_size:
            raise ValueError("The encoder received a window with an unexpected time dimension.")
        features = adapted_sequence.reshape(adapted_sequence.shape[0], -1)
        features = torch.cat((features, parameters), dim=-1)
        return self.network(features)


class FidelityEncoderHead(nn.Module):
    """Fidelity-specific mapping from shared encoder features to latent states."""

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, latent_dim)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.linear(hidden)


class SharedLatentProjector(nn.Module):
    """Common hidden-to-latent projection used across fidelities."""

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            _activation_from_name(activation),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.network(hidden)


class SharedDecoder(nn.Module):
    """Decoder backbone shared across fidelities."""

    def __init__(
        self,
        latent_dim: int,
        parameter_dim: int,
        hidden_dim: int,
        num_hidden_layers: int = 2,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.network = _build_mlp(
            input_dim=latent_dim + parameter_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
        )

    def forward(self, latents: Tensor, parameters: Tensor) -> Tensor:
        if latents.ndim not in {2, 3}:
            raise ValueError("latents must have shape [batch, latent] or [batch, time, latent].")
        if latents.ndim == 2:
            features = torch.cat((latents, parameters), dim=-1)
            return self.network(features)
        batch_size, time_steps, _ = latents.shape
        expanded_parameters = parameters.unsqueeze(1).expand(batch_size, time_steps, -1)
        features = torch.cat((latents, expanded_parameters), dim=-1).reshape(batch_size * time_steps, -1)
        hidden = self.network(features)
        return hidden.reshape(batch_size, time_steps, -1)


class FidelityDecoderHead(nn.Module):
    """Fidelity-specific reconstruction head."""

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.linear(hidden)


class DiscrepancyDecoder(nn.Module):
    """Additive correction term for hierarchical high-fidelity decoding."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            _activation_from_name(activation),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.network(hidden)
