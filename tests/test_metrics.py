from __future__ import annotations

from torch.utils.data import DataLoader

from mf_lnode import LatentNeuralODEModel, LossComposer, collate_windowed_groups
from mf_lnode.evaluation.metrics import (
    compute_per_fidelity_metrics,
    compute_rollout_metrics,
    evaluate_parameter_holdout,
)


def test_evaluation_metrics_compute_expected_keys(windowed_datasets, tiny_config):
    model = LatentNeuralODEModel.from_config(tiny_config.model)
    batch = collate_windowed_groups([windowed_datasets["test"][0], windowed_datasets["test"][1]])
    outputs = model(batch)
    rollout_metrics = compute_rollout_metrics(outputs)
    flattened = compute_per_fidelity_metrics(outputs)
    assert "low" in rollout_metrics
    assert "low/mse" in flattened


def test_evaluate_parameter_holdout_runs(windowed_datasets, tiny_config):
    model = LatentNeuralODEModel.from_config(tiny_config.model)
    loader = DataLoader(
        windowed_datasets["test"],
        batch_size=2,
        shuffle=False,
        collate_fn=collate_windowed_groups,
    )
    metrics = evaluate_parameter_holdout(
        model=model,
        dataloader=loader,
        loss_composer=LossComposer(tiny_config.loss),
        device=tiny_config.training.device,
    )
    assert "total_loss" in metrics
