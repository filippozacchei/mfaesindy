from __future__ import annotations

import pytest

from mf_lnode import (
    ExperimentConfig,
    MultiFidelityScaler,
    MultiFidelityTrajectoryDataset,
    WindowedTrajectoryDataset,
    build_cfd_grid_multifidelity_dataset,
    build_synthetic_multifidelity_dataset,
    split_group_dataset,
)


@pytest.fixture()
def tiny_config(tmp_path):
    config = ExperimentConfig()
    config.seed = 5
    config.synthetic.seed = 5
    config.synthetic.num_paired = 8
    config.synthetic.num_low_only = 2
    config.synthetic.num_high_only = 2
    config.synthetic.observation_dim = 12
    config.synthetic.trajectory_length = 18
    config.data.window_size = 4
    config.data.rollout_horizon = 3
    config.model.window_size = config.data.window_size
    config.training.epochs = 1
    config.training.batch_size = 4
    config.training.output_dir = str(tmp_path / "artifacts")
    config.training.device = "cpu"
    return config


@pytest.fixture()
def synthetic_groups(tiny_config):
    return build_synthetic_multifidelity_dataset(tiny_config.synthetic)


@pytest.fixture()
def scaled_datasets(tiny_config, synthetic_groups):
    train_raw, val_raw, test_raw = split_group_dataset(
        synthetic_groups,
        train_fraction=tiny_config.data.train_fraction,
        val_fraction=tiny_config.data.val_fraction,
        seed=tiny_config.seed,
    )
    scaler = MultiFidelityScaler.fit_from_groups(train_raw.groups)
    train_dataset = MultiFidelityTrajectoryDataset(scaler.transform_groups(train_raw.groups))
    val_dataset = MultiFidelityTrajectoryDataset(scaler.transform_groups(val_raw.groups))
    test_dataset = MultiFidelityTrajectoryDataset(scaler.transform_groups(test_raw.groups))
    tiny_config.model.fidelity_dims = train_dataset.fidelity_dims
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "scaler": scaler,
    }


@pytest.fixture()
def windowed_datasets(tiny_config, scaled_datasets):
    def make(dataset):
        return WindowedTrajectoryDataset(
            group_dataset=dataset,
            window_size=tiny_config.data.window_size,
            rollout_horizon=tiny_config.data.rollout_horizon,
            future_context_shift=tiny_config.data.future_context_shift,
        )

    return {key: make(dataset) if key != "scaler" else dataset for key, dataset in scaled_datasets.items()}


@pytest.fixture()
def tiny_cfd_config(tmp_path):
    config = ExperimentConfig()
    config.seed = 13
    config.cfd_grid.seed = 13
    config.cfd_grid.num_paired = 6
    config.cfd_grid.num_low_only = 1
    config.cfd_grid.num_high_only = 1
    config.cfd_grid.trajectory_length = 16
    config.cfd_grid.time_end = 4.0
    config.cfd_grid.low_grid_shape = (8, 8)
    config.cfd_grid.high_grid_shape = (12, 12)
    config.data.window_size = 4
    config.data.rollout_horizon = 3
    config.model.window_size = config.data.window_size
    config.model.parameter_dim = config.cfd_grid.parameter_dim
    config.model.hierarchical_decoder = False
    config.model.latent_dim = 6
    config.model.adapter_dim = 16
    config.model.encoder_hidden_dim = 32
    config.model.decoder_hidden_dim = 32
    config.model.dynamics_hidden_dim = 32
    config.training.epochs = 1
    config.training.batch_size = 2
    config.training.output_dir = str(tmp_path / "artifacts_cfd")
    config.training.device = "cpu"
    return config


@pytest.fixture()
def cfd_groups(tiny_cfd_config):
    return build_cfd_grid_multifidelity_dataset(tiny_cfd_config.cfd_grid)


@pytest.fixture()
def scaled_cfd_datasets(tiny_cfd_config, cfd_groups):
    train_raw, val_raw, test_raw = split_group_dataset(
        cfd_groups,
        train_fraction=tiny_cfd_config.data.train_fraction,
        val_fraction=tiny_cfd_config.data.val_fraction,
        seed=tiny_cfd_config.seed,
    )
    scaler = MultiFidelityScaler.fit_from_groups(train_raw.groups)
    train_dataset = MultiFidelityTrajectoryDataset(scaler.transform_groups(train_raw.groups))
    val_dataset = MultiFidelityTrajectoryDataset(scaler.transform_groups(val_raw.groups))
    test_dataset = MultiFidelityTrajectoryDataset(scaler.transform_groups(test_raw.groups))
    tiny_cfd_config.model.fidelity_dims = train_dataset.fidelity_dims
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "scaler": scaler,
    }


@pytest.fixture()
def windowed_cfd_datasets(tiny_cfd_config, scaled_cfd_datasets):
    def make(dataset):
        return WindowedTrajectoryDataset(
            group_dataset=dataset,
            window_size=tiny_cfd_config.data.window_size,
            rollout_horizon=tiny_cfd_config.data.rollout_horizon,
            future_context_shift=tiny_cfd_config.data.future_context_shift,
        )

    return {key: make(dataset) if key != "scaler" else dataset for key, dataset in scaled_cfd_datasets.items()}
