from __future__ import annotations

import pytest

from mf_lnode.configs.schema import ExperimentConfig, ModelConfig


def test_experiment_config_defaults_are_constructible():
    config = ExperimentConfig()
    assert config.data.window_size == 6
    assert config.model.latent_dim == 8
    assert config.cfd_grid.high_grid_shape == (32, 32)


def test_hierarchical_decoder_requires_matching_dims():
    with pytest.raises(ValueError):
        ModelConfig(
            fidelity_dims={"low": 2, "high": 3},
            hierarchical_decoder=True,
        )
