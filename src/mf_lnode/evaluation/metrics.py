"""Evaluation metrics for rollout and fidelity-wise performance."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch import Tensor

from mf_lnode.losses.core import LossComposer


def _move_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: _move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [_move_to_device(value, device) for value in batch]
    return batch


def compute_rollout_metrics(outputs: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Compute standard rollout metrics for each fidelity."""

    metrics: dict[str, dict[str, float]] = {}
    for fidelity, fidelity_output in outputs["per_fidelity"].items():
        prediction = fidelity_output["predictions"]
        target = fidelity_output["targets"]
        error = prediction - target
        mse = float(error.pow(2).mean().detach().cpu())
        mae = float(error.abs().mean().detach().cpu())
        relative_l2 = float(
            error.norm().div(target.norm().clamp_min(1e-6)).detach().cpu()
        )
        metrics[fidelity] = {
            "mse": mse,
            "mae": mae,
            "relative_l2": relative_l2,
        }
    return metrics


def compute_per_fidelity_metrics(outputs: dict[str, Any]) -> dict[str, float]:
    """Flatten rollout metrics into a single dictionary keyed by fidelity."""

    flattened: dict[str, float] = {}
    for fidelity, metrics in compute_rollout_metrics(outputs).items():
        for name, value in metrics.items():
            flattened[f"{fidelity}/{name}"] = value
    return flattened


@torch.no_grad()
def evaluate_parameter_holdout(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_composer: LossComposer | None = None,
    device: str | torch.device = "cpu",
    fidelity_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Evaluate a trained model on a parameter-holdout dataloader."""

    torch_device = torch.device(device)
    model.eval()
    aggregate: defaultdict[str, float] = defaultdict(float)
    batches = 0
    for batch in dataloader:
        batch = _move_to_device(batch, torch_device)
        outputs = model(batch)
        if loss_composer is not None:
            total_loss, loss_metrics = loss_composer(outputs, fidelity_weights=fidelity_weights)
            aggregate["total_loss"] += float(total_loss.detach().cpu())
            for key, value in loss_metrics.items():
                aggregate[key] += value
        for key, value in compute_per_fidelity_metrics(outputs).items():
            aggregate[key] += value
        batches += 1
    if batches == 0:
        return {}
    return {key: value / batches for key, value in aggregate.items()}

