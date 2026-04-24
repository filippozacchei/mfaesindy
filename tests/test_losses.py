from __future__ import annotations

from mf_lnode import LatentNeuralODEModel, LossComposer, collate_windowed_groups


def test_loss_composer_returns_scalar_and_metrics(windowed_datasets, tiny_config):
    model = LatentNeuralODEModel.from_config(tiny_config.model)
    batch = collate_windowed_groups([windowed_datasets["train"][0], windowed_datasets["train"][1]])
    outputs = model(batch)
    composer = LossComposer(tiny_config.loss)
    total_loss, metrics = composer(outputs, fidelity_weights={"low": 1.0, "high": 2.0})
    assert total_loss.ndim == 0
    assert total_loss.requires_grad
    assert "rollout_loss" in metrics
    assert "latent_alignment_loss" in metrics
