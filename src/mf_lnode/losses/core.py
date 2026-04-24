"""Composable losses for multi-fidelity latent Neural ODE training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from mf_lnode.configs.schema import LossConfig


def _zero_from_outputs(outputs: dict[str, Any]) -> Tensor:
    per_fidelity = outputs["per_fidelity"]
    if not per_fidelity:
        return torch.tensor(0.0)
    first = next(iter(per_fidelity.values()))
    return first["z0"].new_tensor(0.0)


def rollout_loss(outputs: dict[str, Any]) -> Tensor:
    """Average reconstruction and rollout loss across available fidelities."""

    per_fidelity = outputs["per_fidelity"]
    if not per_fidelity:
        return _zero_from_outputs(outputs)
    losses = [
        F.mse_loss(fidelity_output["predictions"], fidelity_output["targets"])
        for fidelity_output in per_fidelity.values()
    ]
    return torch.stack(losses).mean()


def latent_consistency_loss(outputs: dict[str, Any]) -> Tensor:
    """Match encoded future windows with ODE-propagated latent states."""

    per_fidelity = outputs["per_fidelity"]
    losses: list[Tensor] = []
    for fidelity_output in per_fidelity.values():
        future_latent = fidelity_output["future_latent"]
        if future_latent is None or fidelity_output["latent_trajectory"].shape[1] < 2:
            continue
        predicted_future = fidelity_output["latent_trajectory"][:, 1, :]
        losses.append(F.mse_loss(predicted_future, future_latent))
    if not losses:
        return _zero_from_outputs(outputs)
    return torch.stack(losses).mean()


def paired_latent_alignment_loss(
    outputs: dict[str, Any],
    align_trajectories: bool = True,
) -> Tensor:
    """Align paired latent states and, when possible, paired latent trajectories."""

    per_fidelity = outputs["per_fidelity"]
    fidelities = sorted(per_fidelity)
    if len(fidelities) < 2:
        return _zero_from_outputs(outputs)

    losses: list[Tensor] = []
    for index, first_fidelity in enumerate(fidelities[:-1]):
        first_output = per_fidelity[first_fidelity]
        first_lookup = {
            int(group_id): batch_index
            for batch_index, group_id in enumerate(first_output["group_ids"].detach().cpu().tolist())
        }
        for second_fidelity in fidelities[index + 1 :]:
            second_output = per_fidelity[second_fidelity]
            second_lookup = {
                int(group_id): batch_index
                for batch_index, group_id in enumerate(second_output["group_ids"].detach().cpu().tolist())
            }
            common_group_ids = sorted(set(first_lookup) & set(second_lookup))
            if not common_group_ids:
                continue

            first_indices = first_output["z0"].new_tensor(
                [first_lookup[group_id] for group_id in common_group_ids],
                dtype=torch.long,
            )
            second_indices = second_output["z0"].new_tensor(
                [second_lookup[group_id] for group_id in common_group_ids],
                dtype=torch.long,
            )
            first_z0 = first_output["z0"].index_select(0, first_indices)
            second_z0 = second_output["z0"].index_select(0, second_indices)
            pair_loss = F.mse_loss(first_z0, second_z0)

            if align_trajectories:
                first_times = first_output["target_times"].index_select(0, first_indices)
                second_times = second_output["target_times"].index_select(0, second_indices)
                if first_times.shape == second_times.shape and torch.allclose(first_times, second_times):
                    first_trajectory = first_output["latent_trajectory"].index_select(0, first_indices)
                    second_trajectory = second_output["latent_trajectory"].index_select(0, second_indices)
                    shared_steps = min(first_trajectory.shape[1], second_trajectory.shape[1])
                    pair_loss = pair_loss + F.mse_loss(
                        first_trajectory[:, :shared_steps, :],
                        second_trajectory[:, :shared_steps, :],
                    )
            losses.append(pair_loss)

    if not losses:
        return _zero_from_outputs(outputs)
    return torch.stack(losses).mean()


def multifidelity_reconstruction_loss(
    outputs: dict[str, Any],
    fidelity_weights: dict[str, float] | None = None,
) -> Tensor:
    """Weighted reconstruction loss over fidelities."""

    per_fidelity = outputs["per_fidelity"]
    if not per_fidelity:
        return _zero_from_outputs(outputs)
    fidelity_weights = fidelity_weights or {}
    weighted_losses: list[Tensor] = []
    weights: list[Tensor] = []
    for fidelity, fidelity_output in per_fidelity.items():
        weight = fidelity_weights.get(fidelity, 1.0)
        weighted_losses.append(
            F.mse_loss(fidelity_output["predictions"], fidelity_output["targets"]) * weight
        )
        weights.append(fidelity_output["z0"].new_tensor(weight))
    return torch.stack(weighted_losses).sum() / torch.stack(weights).sum().clamp_min(1e-6)


def discrepancy_loss(outputs: dict[str, Any]) -> Tensor:
    """Regularize additive discrepancy corrections in hierarchical decoding."""

    per_fidelity = outputs["per_fidelity"]
    losses = [
        fidelity_output["discrepancy"].pow(2).mean()
        for fidelity_output in per_fidelity.values()
        if fidelity_output["discrepancy"] is not None
    ]
    if not losses:
        return _zero_from_outputs(outputs)
    return torch.stack(losses).mean()


def latent_regularization_loss(outputs: dict[str, Any]) -> Tensor:
    """Simple latent norm and smoothness regularization."""

    per_fidelity = outputs["per_fidelity"]
    if not per_fidelity:
        return _zero_from_outputs(outputs)
    losses: list[Tensor] = []
    for fidelity_output in per_fidelity.values():
        z0_penalty = fidelity_output["z0"].pow(2).mean()
        trajectory = fidelity_output["latent_trajectory"]
        if trajectory.shape[1] > 1:
            smoothness = (trajectory[:, 1:, :] - trajectory[:, :-1, :]).pow(2).mean()
            losses.append(z0_penalty + smoothness)
        else:
            losses.append(z0_penalty)
    return torch.stack(losses).mean()


class LossComposer:
    """Combine all configured training losses into a single scalar objective."""

    def __init__(self, config: LossConfig) -> None:
        self.config = config

    def __call__(
        self,
        outputs: dict[str, Any],
        fidelity_weights: dict[str, float] | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        roll = rollout_loss(outputs)
        latent = latent_consistency_loss(outputs)
        latent_alignment = paired_latent_alignment_loss(
            outputs,
            align_trajectories=self.config.align_latent_trajectories,
        )
        multifidelity = multifidelity_reconstruction_loss(outputs, fidelity_weights=fidelity_weights)
        discrepancy = discrepancy_loss(outputs)
        regularization = latent_regularization_loss(outputs)

        total = (
            self.config.rollout_weight * roll
            + self.config.latent_consistency_weight * latent
            + self.config.latent_alignment_weight * latent_alignment
            + self.config.multifidelity_weight * multifidelity
            + self.config.discrepancy_weight * discrepancy
            + self.config.regularization_weight * regularization
        )
        metrics = {
            "rollout_loss": float(roll.detach().cpu()),
            "latent_consistency_loss": float(latent.detach().cpu()),
            "latent_alignment_loss": float(latent_alignment.detach().cpu()),
            "multifidelity_loss": float(multifidelity.detach().cpu()),
            "discrepancy_loss": float(discrepancy.detach().cpu()),
            "regularization_loss": float(regularization.detach().cpu()),
        }
        return total, metrics
