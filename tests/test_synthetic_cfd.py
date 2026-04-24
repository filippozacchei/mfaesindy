from __future__ import annotations

import torch

from mf_lnode.data.synthetic_cfd import (
    build_cfd_grid_multifidelity_dataset,
    simulate_cfd_like_field,
)


def test_simulate_cfd_like_field_returns_expected_shape():
    trajectory = simulate_cfd_like_field(
        parameter=torch.tensor([1.0, 0.9]),
        times=torch.linspace(0.0, 1.0, 10),
        grid_shape=(8, 12),
        fidelity="high",
    )
    assert trajectory.shape == (10, 8, 12)


def test_build_cfd_grid_multifidelity_dataset_tracks_grid_metadata(tiny_cfd_config):
    groups = build_cfd_grid_multifidelity_dataset(tiny_cfd_config.cfd_grid)
    expected = (
        tiny_cfd_config.cfd_grid.num_paired
        + tiny_cfd_config.cfd_grid.num_low_only
        + tiny_cfd_config.cfd_grid.num_high_only
    )
    assert len(groups) == expected

    paired = next(group for group in groups if {"low", "high"}.issubset(group.trajectories))
    low = paired.trajectories["low"]
    high = paired.trajectories["high"]

    assert low.observation_dim == tiny_cfd_config.cfd_grid.low_grid_shape[0] * tiny_cfd_config.cfd_grid.low_grid_shape[1]
    assert high.observation_dim == tiny_cfd_config.cfd_grid.high_grid_shape[0] * tiny_cfd_config.cfd_grid.high_grid_shape[1]
    assert low.metadata["grid_height"] == tiny_cfd_config.cfd_grid.low_grid_shape[0]
    assert low.metadata["grid_width"] == tiny_cfd_config.cfd_grid.low_grid_shape[1]
    assert high.metadata["grid_height"] == tiny_cfd_config.cfd_grid.high_grid_shape[0]
    assert high.metadata["grid_width"] == tiny_cfd_config.cfd_grid.high_grid_shape[1]
