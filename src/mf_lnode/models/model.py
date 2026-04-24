"""Top-level latent Neural ODE model for multi-fidelity trajectory learning."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from mf_lnode.configs.schema import ModelConfig
from mf_lnode.dynamics.neural_ode import LatentDynamics, NeuralODESolver, ParameterizedVectorField
from mf_lnode.models.autoencoder import (
    DiscrepancyDecoder,
    FidelityDecoderHead,
    FidelityEncoderHead,
    SharedLatentProjector,
    SharedDecoder,
    SharedEncoder,
)


class LatentNeuralODEModel(nn.Module):
    """Shared latent Neural ODE with fidelity-specific autoencoding heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.fidelities = tuple(sorted(config.fidelity_dims))

        self.encoder_adapters = nn.ModuleDict(
            {
                fidelity: nn.Linear(observation_dim + 1, config.adapter_dim)
                for fidelity, observation_dim in config.fidelity_dims.items()
            }
        )
        self.encoder_heads = (
            nn.ModuleDict(
                {
                    fidelity: FidelityEncoderHead(config.encoder_hidden_dim, config.latent_dim)
                    for fidelity in self.fidelities
                }
            )
            if not config.shared_latent_projection
            else None
        )
        self.shared_latent_projector = (
            SharedLatentProjector(
                hidden_dim=config.encoder_hidden_dim,
                latent_dim=config.latent_dim,
                activation=config.activation,
            )
            if config.shared_latent_projection
            else None
        )
        self.shared_encoder = (
            SharedEncoder(
                window_size=config.window_size,
                adapter_dim=config.adapter_dim,
                parameter_dim=config.parameter_dim,
                hidden_dim=config.encoder_hidden_dim,
                num_hidden_layers=config.num_hidden_layers,
                activation=config.activation,
            )
            if config.shared_encoder_backbone
            else None
        )
        self.encoder_backbones = (
            nn.ModuleDict(
                {
                    fidelity: SharedEncoder(
                        window_size=config.window_size,
                        adapter_dim=config.adapter_dim,
                        parameter_dim=config.parameter_dim,
                        hidden_dim=config.encoder_hidden_dim,
                        num_hidden_layers=config.num_hidden_layers,
                        activation=config.activation,
                    )
                    for fidelity in self.fidelities
                }
            )
            if not config.shared_encoder_backbone
            else None
        )

        self.decoder_heads = nn.ModuleDict(
            {
                fidelity: FidelityDecoderHead(config.decoder_hidden_dim, observation_dim)
                for fidelity, observation_dim in config.fidelity_dims.items()
            }
        )
        self.shared_decoder = (
            SharedDecoder(
                latent_dim=config.latent_dim,
                parameter_dim=config.parameter_dim,
                hidden_dim=config.decoder_hidden_dim,
                num_hidden_layers=config.num_hidden_layers,
                activation=config.activation,
            )
            if config.shared_decoder_backbone
            else None
        )
        self.decoder_backbones = (
            nn.ModuleDict(
                {
                    fidelity: SharedDecoder(
                        latent_dim=config.latent_dim,
                        parameter_dim=config.parameter_dim,
                        hidden_dim=config.decoder_hidden_dim,
                        num_hidden_layers=config.num_hidden_layers,
                        activation=config.activation,
                    )
                    for fidelity in self.fidelities
                }
            )
            if not config.shared_decoder_backbone
            else None
        )
        self.discrepancy_decoder = (
            DiscrepancyDecoder(
                hidden_dim=config.decoder_hidden_dim,
                output_dim=config.fidelity_dims[config.hierarchy_high_fidelity],
                activation=config.activation,
            )
            if config.hierarchical_decoder
            else None
        )

        vector_field = ParameterizedVectorField(
            latent_dim=config.latent_dim,
            parameter_dim=config.parameter_dim,
            hidden_dim=config.dynamics_hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            activation=config.activation,
            conditioning=config.conditioning,
        )
        self.dynamics = LatentDynamics(vector_field)
        self.solver = NeuralODESolver(
            backend=config.solver_backend,
            steps_per_interval=config.solver_steps_per_interval,
        )

    @classmethod
    def from_config(cls, config: ModelConfig) -> "LatentNeuralODEModel":
        """Construct the model directly from a validated config object."""

        return cls(config)

    def _get_encoder_backbone(self, fidelity: str) -> SharedEncoder:
        if self.shared_encoder is not None:
            return self.shared_encoder
        assert self.encoder_backbones is not None
        return self.encoder_backbones[fidelity]

    def _get_decoder_backbone(self, fidelity: str) -> SharedDecoder:
        if self.shared_decoder is not None:
            return self.shared_decoder
        assert self.decoder_backbones is not None
        return self.decoder_backbones[fidelity]

    def encode(
        self,
        fidelity: str,
        context_times: Tensor,
        context_observations: Tensor,
        parameters: Tensor,
    ) -> Tensor:
        """Encode a temporal observation window into a latent initial condition."""

        relative_times = context_times - context_times[:, -1:].expand_as(context_times)
        features = torch.cat((context_observations, relative_times.unsqueeze(-1)), dim=-1)
        adapted = self.encoder_adapters[fidelity](features)
        hidden = self._get_encoder_backbone(fidelity)(adapted, parameters)
        if self.shared_latent_projector is not None:
            return self.shared_latent_projector(hidden)
        assert self.encoder_heads is not None
        return self.encoder_heads[fidelity](hidden)

    def decode(
        self,
        fidelity: str,
        latents: Tensor,
        parameters: Tensor,
    ) -> dict[str, Tensor | None]:
        """Decode latent states into observations at the requested fidelity."""

        decoder_backbone = self._get_decoder_backbone(fidelity)
        hidden = decoder_backbone(latents, parameters)
        base_prediction = self.decoder_heads[fidelity](hidden)

        if not self.config.hierarchical_decoder or fidelity != self.config.hierarchy_high_fidelity:
            return {
                "prediction": base_prediction,
                "low_prediction": None,
                "discrepancy": None,
            }

        low_fidelity = self.config.hierarchy_low_fidelity
        low_hidden = self._get_decoder_backbone(low_fidelity)(latents, parameters)
        low_prediction = self.decoder_heads[low_fidelity](low_hidden)
        assert self.discrepancy_decoder is not None
        discrepancy = self.discrepancy_decoder(hidden)
        return {
            "prediction": low_prediction + discrepancy,
            "low_prediction": low_prediction,
            "discrepancy": discrepancy,
        }

    def rollout_fidelity(
        self,
        fidelity: str,
        context_times: Tensor,
        context_observations: Tensor,
        target_times: Tensor,
        parameters: Tensor,
        future_context_times: Tensor | None = None,
        future_context_observations: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        """Encode, roll out latent dynamics, and decode a single fidelity batch."""

        z0 = self.encode(fidelity, context_times, context_observations, parameters)
        latent_trajectory = self.solver.integrate(self.dynamics, z0, target_times, parameters)
        decoded = self.decode(fidelity, latent_trajectory, parameters)
        future_latent = None
        if future_context_times is not None and future_context_observations is not None:
            future_latent = self.encode(
                fidelity,
                future_context_times,
                future_context_observations,
                parameters,
            )
        return {
            "z0": z0,
            "latent_trajectory": latent_trajectory,
            "prediction": decoded["prediction"],
            "low_prediction": decoded["low_prediction"],
            "discrepancy": decoded["discrepancy"],
            "future_latent": future_latent,
        }

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run the full model on a collated grouped batch."""

        outputs: dict[str, Any] = {"per_fidelity": {}}
        for fidelity, tensors in batch["windows"].items():
            fidelity_output = self.rollout_fidelity(
                fidelity=fidelity,
                context_times=tensors["context_times"],
                context_observations=tensors["context_observations"],
                target_times=tensors["target_times"],
                parameters=tensors["parameters"],
                future_context_times=tensors["future_context_times"],
                future_context_observations=tensors["future_context_observations"],
            )
            outputs["per_fidelity"][fidelity] = {
                "z0": fidelity_output["z0"],
                "latent_trajectory": fidelity_output["latent_trajectory"],
                "predictions": fidelity_output["prediction"],
                "low_predictions": fidelity_output["low_prediction"],
                "discrepancy": fidelity_output["discrepancy"],
                "future_latent": fidelity_output["future_latent"],
                "targets": tensors["target_observations"],
                "target_times": tensors["target_times"],
                "parameters": tensors["parameters"],
                "group_ids": tensors["group_ids"],
                "pairing_ids": tensors["pairing_ids"],
            }
        return outputs

    def rollout(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Alias for the batch forward pass used by the high-level API."""

        return self.forward(batch)
