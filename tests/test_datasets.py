from __future__ import annotations

from mf_lnode.data.datasets import (
    MultiFidelityTrajectoryDataset,
    WindowedTrajectoryDataset,
    collate_windowed_groups,
    split_group_dataset,
    split_group_dataset_by_parameter,
)


def test_group_dataset_exposes_fidelity_dimensions(synthetic_groups, tiny_config):
    dataset = MultiFidelityTrajectoryDataset(synthetic_groups)
    expected_dim = tiny_config.synthetic.observation_dim
    assert dataset.fidelity_dims == {"high": expected_dim, "low": expected_dim}


def test_windowed_dataset_and_collate_produce_per_fidelity_batches(synthetic_groups, tiny_config):
    dataset = MultiFidelityTrajectoryDataset(synthetic_groups)
    windowed = WindowedTrajectoryDataset(
        group_dataset=dataset,
        window_size=tiny_config.data.window_size,
        rollout_horizon=tiny_config.data.rollout_horizon,
        future_context_shift=tiny_config.data.future_context_shift,
    )
    batch = collate_windowed_groups([windowed[0], windowed[1]])
    assert "low" in batch["windows"]
    assert "high" in batch["windows"]
    assert batch["windows"]["low"]["context_observations"].ndim == 3


def test_group_splitting_preserves_total_size(synthetic_groups):
    train, val, test = split_group_dataset(synthetic_groups, seed=3)
    total = len(train) + len(val) + len(test)
    assert total == len(synthetic_groups)


def test_parameter_splitting_preserves_total_size(synthetic_groups):
    train, val, test = split_group_dataset_by_parameter(synthetic_groups, seed=3)
    total = len(train) + len(val) + len(test)
    assert total == len(synthetic_groups)
