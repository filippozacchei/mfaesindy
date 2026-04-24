from __future__ import annotations

from mf_lnode import LatentNeuralODEModel, collate_windowed_groups


def test_latent_neural_ode_model_runs_forward(windowed_datasets, tiny_config):
    model = LatentNeuralODEModel.from_config(tiny_config.model)
    dataset = windowed_datasets["train"]
    batch = collate_windowed_groups([dataset[0], dataset[1]])
    outputs = model(batch)
    assert "low" in outputs["per_fidelity"]
    assert "high" in outputs["per_fidelity"]
    assert outputs["per_fidelity"]["low"]["predictions"].shape[-1] == 2

