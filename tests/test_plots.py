from __future__ import annotations

from pathlib import Path

import torch

from mf_lnode import LatentNeuralODEModel, collate_windowed_groups, split_group_dataset
from mf_lnode.evaluation.plots import (
    plot_architecture_breakdown,
    plot_dataset_overview,
    plot_fidelity_gap,
    plot_full_trajectory_rollout,
    plot_latent_diagnostics,
    plot_measurement_overview,
    plot_method_overview,
    plot_parameter_coverage,
    plot_representative_rollout,
    plot_rollout_error_profile,
    plot_test_problem_summary,
    plot_test_metrics,
    plot_training_strategy_overview,
    plot_training_history,
)


def test_plotting_utilities_emit_files(
    synthetic_groups,
    windowed_datasets,
    scaled_datasets,
    tiny_config,
    tmp_path,
):
    model = LatentNeuralODEModel.from_config(tiny_config.model)
    batch = collate_windowed_groups([windowed_datasets["test"][0]])
    outputs = model(batch)
    train_raw, val_raw, test_raw = split_group_dataset(synthetic_groups, seed=tiny_config.seed)
    history = [
        {
            "epoch": 1.0,
            "train/total_loss": 1.2,
            "train/rollout_loss": 0.8,
            "train/multifidelity_loss": 0.6,
            "train/latent_consistency_loss": 0.2,
            "train/latent_alignment_loss": 0.15,
            "train/discrepancy_loss": 0.05,
            "train/regularization_loss": 0.01,
            "val/total_loss": 1.0,
            "val/rollout_loss": 0.7,
            "train/high/mse": 0.5,
            "train/low/mse": 0.6,
            "val/high/mse": 0.4,
            "val/low/mse": 0.5,
        }
    ]
    fidelity_weight_history = [{"low": 1.0, "high": 0.5}]
    metrics = {
        "high/mse": 0.2,
        "high/mae": 0.3,
        "high/relative_l2": 0.4,
        "low/mse": 0.5,
        "low/mae": 0.6,
        "low/relative_l2": 0.7,
        "latent_alignment_loss": 0.15,
    }
    full_rollout = {
        "low": {
            "pairing_id": "group-0000",
            "parameter": torch.tensor([0.1, 1.1]),
            "full_times": torch.linspace(0.0, 1.0, 8),
            "full_targets": torch.stack((torch.linspace(0.0, 1.0, 8), torch.linspace(1.0, 0.0, 8)), dim=-1),
            "context_times": torch.linspace(0.0, 0.3, 3),
            "context_targets": torch.tensor([[0.0, 1.0], [0.15, 0.85], [0.3, 0.7]]),
            "rollout_times": torch.linspace(0.3, 1.0, 6),
            "rollout_targets": torch.tensor(
                [[0.3, 0.7], [0.45, 0.55], [0.6, 0.4], [0.75, 0.25], [0.9, 0.1], [1.0, 0.0]]
            ),
            "prediction": torch.tensor(
                [[0.28, 0.72], [0.44, 0.56], [0.61, 0.39], [0.73, 0.27], [0.88, 0.12], [0.98, 0.02]]
            ),
            "latent_trajectory": torch.randn(6, tiny_config.model.latent_dim),
            "z0": torch.randn(tiny_config.model.latent_dim),
        },
        "high": {
            "pairing_id": "group-0000",
            "parameter": torch.tensor([0.1, 1.1]),
            "full_times": torch.linspace(0.0, 1.0, 8),
            "full_targets": torch.stack((torch.linspace(0.0, 1.1, 8), torch.linspace(1.0, -0.1, 8)), dim=-1),
            "context_times": torch.linspace(0.0, 0.3, 3),
            "context_targets": torch.tensor([[0.0, 1.0], [0.16, 0.84], [0.33, 0.67]]),
            "rollout_times": torch.linspace(0.3, 1.0, 6),
            "rollout_targets": torch.tensor(
                [[0.33, 0.67], [0.5, 0.5], [0.67, 0.33], [0.83, 0.17], [0.98, 0.02], [1.1, -0.1]]
            ),
            "prediction": torch.tensor(
                [[0.31, 0.69], [0.49, 0.51], [0.69, 0.31], [0.81, 0.19], [0.97, 0.03], [1.08, -0.08]]
            ),
            "latent_trajectory": torch.randn(6, tiny_config.model.latent_dim),
            "z0": torch.randn(tiny_config.model.latent_dim),
        },
    }

    paths = [
        plot_test_problem_summary(test_raw.groups, tmp_path / "test_problem_summary.png"),
        plot_measurement_overview(test_raw.groups, tmp_path / "measurement_overview.png"),
        plot_method_overview(tiny_config.model, tmp_path / "method_overview.png"),
        plot_architecture_breakdown(tiny_config.model, tmp_path / "architecture_breakdown.png"),
        plot_training_strategy_overview(
            tiny_config,
            history,
            fidelity_weight_history,
            tmp_path / "training_strategy.png",
        ),
        plot_dataset_overview(test_raw.groups, tmp_path / "dataset_overview.png"),
        plot_parameter_coverage(
            {"train": train_raw.groups, "val": val_raw.groups, "test": test_raw.groups},
            tmp_path / "parameter_coverage.png",
        ),
        plot_training_history(
            history,
            tmp_path / "training_history.png",
            fidelity_weight_history=fidelity_weight_history,
        ),
        plot_representative_rollout(
            batch,
            outputs,
            scaled_datasets["scaler"],
            tmp_path / "representative_rollout.png",
        ),
        plot_full_trajectory_rollout(full_rollout, tmp_path / "full_trajectory_rollout.png"),
        plot_rollout_error_profile(full_rollout, tmp_path / "rollout_error_profile.png"),
        plot_fidelity_gap(full_rollout, tmp_path / "fidelity_gap.png"),
        plot_latent_diagnostics(full_rollout, tmp_path / "latent_diagnostics.png"),
        plot_test_metrics(metrics, tmp_path / "test_metrics.png"),
    ]

    for path in paths:
        assert Path(path).exists()
